// fneq-lift: lift one machine function with remill and lower it to a closed,
// poison-free LLVM function that Alive2 can compare against another one.
//
// Input: a JSON "model spec" written by the Python orchestrator (see
// fneq/alive_backend.py for the schema). Output: textual LLVM IR containing
//
//   define {outs...} @<name>(ptr %M, <register inputs>, <stack-arg inputs>)
//
// Memory model (all soundness-relevant decisions live here):
//   * %M is a flat byte array covering the whole address space (x86: 2^32,
//     x86-64: 2^47 canonical user space). Every machine address is a valid
//     index, so a memory access can never be UB in the lowered IR.
//   * The function's own stack frame [esp0-below, esp0+above) is a local
//     alloca initialised from %M. Its final contents are not observable, so
//     spills, locals and outgoing argument slots never count as effects.
//     Stack provenance is tracked through a marker value (@fneq.esp0) rather
//     than through address ranges; addresses that do not derive from esp0 are
//     treated as non-stack memory (they cannot point into a frame that does
//     not exist yet when the function is entered).
//   * Read-only data ranges supplied in the spec are constant tables.
//   * External callees become calls to uninterpreted functions @ext.<name>
//     taking their arguments by value and returning their results plus the
//     clobbered caller-saved registers.
//
// UB policy: machine code has no undefined behaviour, so after lifting every
// poison-generating flag, UB-implying attribute and metadata is dropped,
// shifts are made total, divisions are guarded (a division fault becomes an
// explicit output), lifted locals are zero-initialised and the final IR is
// scanned for undef/poison/unreachable. Any violation is a hard error.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/Signals.h>
#include <remill/Arch/Arch.h>
#include <remill/Arch/Name.h>
#include <remill/BC/ABI.h>
#include <remill/BC/IntrinsicTable.h>
#include <remill/BC/Lifter.h>
#include <remill/BC/TraceLifter.h>
#include <remill/BC/Util.h>
#include <remill/OS/OS.h>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

DEFINE_string(spec, "", "Path to the JSON model spec.");
DEFINE_string(out, "", "Path of the lowered LLVM IR to write.");
DEFINE_string(pre_out, "", "Optional path for the IR right after lifting.");
DEFINE_string(finalize, "", "Finalize mode: rewrite fault markers in this IR file to unreachable and write the result to --out (no other change).");

using namespace llvm;

namespace {

// ---------------------------------------------------------------- diagnostics

json::Array g_unsupported;  // reasons that make the result UNKNOWN
json::Array g_notes;
json::Array g_escapes;      // esp0-relative offsets of escaping stack addresses
bool g_escape_unknown = false;  // an escaping stack address with a non-constant offset

[[noreturn]] void Finish(bool ok, const std::string &error = "") {
  json::Object o;
  o["ok"] = ok;
  o["unsupported"] = std::move(g_unsupported);
  o["notes"] = std::move(g_notes);
  if (!g_escapes.empty() || g_escape_unknown) {
    o["escape_offsets"] = std::move(g_escapes);
    o["escape_unknown"] = g_escape_unknown;
  }
  if (!error.empty()) o["error"] = error;
  llvm::outs() << json::Value(std::move(o)) << "\n";
  llvm::outs().flush();
  std::exit(ok ? 0 : 4);
}

void Unsupported(const std::string &why) {
  g_unsupported.push_back(why);
}

[[noreturn]] void Fail(const std::string &why) {
  Finish(false, why);
}

// ----------------------------------------------------------------------- spec

struct ArgLoc {
  std::string reg;     // register name, or empty for a stack slot
  int64_t offset = 0;  // stack slot offset relative to the callee-entry SP
  unsigned size = 0;   // bytes
};

struct Callee {
  std::string name;
  std::vector<ArgLoc> args;
  std::vector<std::string> results;  // registers receiving results/clobbers
  uint64_t pop = 0;                  // bytes popped in addition to the RA
};

struct Region {
  uint64_t addr;
  std::vector<uint8_t> bytes;
};

struct Spec {
  std::string arch, os, name = "fneq_fn";
  uint64_t entry = 0;
  std::map<uint64_t, uint8_t> code;
  std::vector<Region> rodata;
  std::vector<std::string> in_regs;
  std::vector<ArgLoc> stack_args;  // offsets relative to esp0
  std::vector<std::string> out_regs;
  std::map<std::string, uint64_t> fixed_regs;  // registers pinned to constants
  uint64_t esp0 = 0, ret_addr = 0, frame_below = 0, frame_above = 0;
  std::map<uint64_t, Callee> callees;
  // Model of indirect calls (__remill_function_call): an uninterpreted function
  // of the target value followed by `args`.
  bool has_indirect = false;
  Callee indirect;
  // A stack address escapes (decided by a first lifting attempt): the frame is
  // visible to callees at its machine addresses.
  bool frame_visible = false;
  // Part of the frame the callees see, relative to esp0: [window_lo, window_hi).
  // Defaults to the whole frame.
  bool has_window = false;
  int64_t window_lo = 0, window_hi = 0;
  // Global mappings: addresses in [from, from+size) are translated to `to`.
  struct Remap { uint64_t from, to, size; };
  std::vector<Remap> remap;
  // Pointer values crossing the function boundary (call arguments, returned
  // registers) translated into side-A space: mapped globals and constant data.
  std::vector<Remap> value_remap;
  std::set<std::string> value_remap_outputs;
};

uint64_t GetU64(const json::Object &o, StringRef key) {
  if (auto v = o.getInteger(key)) return static_cast<uint64_t>(*v);
  if (auto s = o.getString(key)) {
    uint64_t r;
    if (!s->getAsInteger(0, r)) return r;
  }
  Fail(("spec: missing/invalid integer field '" + key + "'").str());
}

std::vector<uint8_t> Unhex(StringRef hex) {
  if (hex.size() % 2) Fail("spec: odd-length hex string");
  std::vector<uint8_t> out;
  for (size_t i = 0; i < hex.size(); i += 2) {
    unsigned v;
    if (hex.substr(i, 2).getAsInteger(16, v)) Fail("spec: bad hex string");
    out.push_back(static_cast<uint8_t>(v));
  }
  return out;
}

ArgLoc ParseArgLoc(const json::Value &v) {
  auto *o = v.getAsObject();
  if (!o) Fail("spec: argument location must be an object");
  ArgLoc a;
  if (auto r = o->getString("reg")) a.reg = r->upper();
  if (auto off = o->getInteger("offset")) a.offset = *off;
  if (auto sz = o->getInteger("size")) a.size = static_cast<unsigned>(*sz);
  if (a.reg.empty() && !a.size) Fail("spec: stack argument without size");
  return a;
}

Spec LoadSpec(const std::string &path) {
  auto buf = MemoryBuffer::getFile(path);
  if (!buf) Fail("cannot read spec file " + path);
  auto parsed = json::parse((*buf)->getBuffer());
  if (!parsed) Fail("spec is not valid JSON: " + toString(parsed.takeError()));
  auto *o = parsed->getAsObject();
  if (!o) Fail("spec must be a JSON object");

  Spec s;
  s.arch = o->getString("arch").value_or("").str();
  s.os = o->getString("os").value_or("linux").str();
  if (auto n = o->getString("name")) s.name = n->str();
  s.entry = GetU64(*o, "entry");
  s.esp0 = GetU64(*o, "esp0");
  s.ret_addr = GetU64(*o, "ret_addr");
  s.frame_below = GetU64(*o, "frame_below");
  s.frame_above = GetU64(*o, "frame_above");

  if (auto *code = o->getArray("code")) {
    for (auto &c : *code) {
      auto *co = c.getAsObject();
      uint64_t addr = GetU64(*co, "addr");
      auto bytes = Unhex(co->getString("hex").value_or(""));
      for (size_t i = 0; i < bytes.size(); ++i) s.code[addr + i] = bytes[i];
    }
  }
  if (auto *ro = o->getArray("rodata")) {
    for (auto &c : *ro) {
      auto *co = c.getAsObject();
      s.rodata.push_back({GetU64(*co, "addr"),
                          Unhex(co->getString("hex").value_or(""))});
    }
  }
  if (auto *a = o->getArray("in_regs"))
    for (auto &r : *a) s.in_regs.push_back(r.getAsString()->upper());
  if (auto *a = o->getArray("out_regs"))
    for (auto &r : *a) s.out_regs.push_back(r.getAsString()->upper());
  if (auto *a = o->getArray("stack_args"))
    for (auto &r : *a) s.stack_args.push_back(ParseArgLoc(r));
  if (auto *f = o->getObject("fixed_regs"))
    for (auto &kv : *f)
      s.fixed_regs[kv.first.str()] = static_cast<uint64_t>(*kv.second.getAsInteger());
  if (auto *cs = o->getArray("callees")) {
    for (auto &c : *cs) {
      auto *co = c.getAsObject();
      Callee callee;
      callee.name = co->getString("name").value_or("").str();
      callee.pop = co->getInteger("pop").value_or(0);
      if (auto *args = co->getArray("args"))
        for (auto &a : *args) callee.args.push_back(ParseArgLoc(a));
      if (auto *res = co->getArray("results"))
        for (auto &r : *res) callee.results.push_back(r.getAsString()->upper());
      if (co->getBoolean("indirect").value_or(false)) {
        s.has_indirect = true;
        s.indirect = callee;  // all indirect call sites share one model
        continue;
      }
      s.callees[GetU64(*co, "addr")] = callee;
    }
  }
  s.frame_visible = o->getBoolean("frame_visible").value_or(false);
  if (auto lo = o->getInteger("frame_window_lo"), hi = o->getInteger("frame_window_hi"); lo && hi) {
    s.has_window = true;
    s.window_lo = *lo;
    s.window_hi = *hi;
  }
  if (auto *rm = o->getArray("remap")) {
    for (auto &r : *rm) {
      auto *ro = r.getAsObject();
      s.remap.push_back({GetU64(*ro, "from"), GetU64(*ro, "to"), GetU64(*ro, "size")});
    }
  }
  if (auto *rm = o->getArray("value_remap")) {
    for (auto &r : *rm) {
      auto *ro = r.getAsObject();
      s.value_remap.push_back({GetU64(*ro, "from"), GetU64(*ro, "to"), GetU64(*ro, "size")});
    }
  }
  if (auto *a = o->getArray("value_remap_outputs"))
    for (auto &r : *a) s.value_remap_outputs.insert(r.getAsString()->upper());
  if (s.arch.empty() || s.code.empty()) Fail("spec: arch and code are required");
  return s;
}

// -------------------------------------------------------------- trace manager

struct Manager : remill::TraceManager {
  const remill::Arch *arch;
  Module *module;
  const Spec &spec;
  std::unordered_map<uint64_t, Function *> traces;

  Manager(const remill::Arch *a, Module *m, const Spec &s)
      : arch(a), module(m), spec(s) {}

  std::string TraceName(uint64_t addr) override {
    return "fneq_trace_" + utohexstr(addr);
  }

  Function *External(uint64_t addr) {
    auto name = "fneq_callee_" + utohexstr(addr);
    if (auto *f = module->getFunction(name)) return f;
    return arch->DeclareLiftedFunction(name, module);
  }

  void SetLiftedTraceDefinition(uint64_t addr, Function *fn) override {
    traces[addr] = fn;
  }

  Function *GetLiftedTraceDeclaration(uint64_t addr) override {
    if (spec.callees.count(addr)) return External(addr);
    return nullptr;
  }

  Function *GetLiftedTraceDefinition(uint64_t addr) override {
    if (spec.callees.count(addr)) return External(addr);
    return nullptr;
  }

  bool TryReadExecutableByte(uint64_t addr, uint8_t *byte) override {
    auto it = spec.code.find(addr);
    if (it == spec.code.end()) return false;
    if (byte) *byte = it->second;
    return true;
  }
};

// ------------------------------------------------------------------- helpers

void RunPipeline(Module &M, StringRef pipeline) {
  // Our own IR edits must leave valid IR; LLVM passes may crash on invalid IR.
  std::string why;
  raw_string_ostream os(why);
  if (verifyModule(M, &os))
    Fail("internal error: invalid IR before '" + pipeline.str() + "': " + why.substr(0, 600));
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager MPM;
  if (auto err = PB.parsePassPipeline(MPM, pipeline))
    Fail("bad pass pipeline: " + toString(std::move(err)));
  MPM.run(M, MAM);
}

constexpr const char *kCleanup =
    "function(sroa,early-cse,instcombine<no-verify-fixpoint>,simplifycfg,sroa,mem2reg,"
    "instcombine<no-verify-fixpoint>,early-cse,gvn,simplifycfg,adce)";

Function *DeclarePure(Module &M, StringRef name, Type *ret,
                      ArrayRef<Type *> params) {
  auto *fn = M.getFunction(name);
  if (!fn) {
    fn = Function::Create(FunctionType::get(ret, params, false),
                          GlobalValue::ExternalLinkage, name, M);
    fn->setDoesNotAccessMemory();
    fn->setDoesNotThrow();
    fn->setWillReturn();
  }
  return fn;
}

// Memory accesses before lowering. Reads are pure functions of the abstract
// memory state; writes change it. Using inaccessible memory lets the
// optimizer reorder accesses to the State alloca around them while keeping
// the relative order of the memory operations themselves.
Function *MemRead(Module &M, unsigned bits, Type *addr_ty) {
  auto name = "fneq.read." + std::to_string(bits);
  auto *fn = M.getFunction(name);
  if (!fn) {
    auto &C = M.getContext();
    fn = Function::Create(
        FunctionType::get(IntegerType::get(C, bits), {addr_ty}, false),
        GlobalValue::ExternalLinkage, name, M);
    fn->setMemoryEffects(MemoryEffects::inaccessibleMemOnly(ModRefInfo::Ref));
    fn->setDoesNotThrow();
    fn->setWillReturn();
  }
  return fn;
}

Function *MemWrite(Module &M, unsigned bits, Type *addr_ty) {
  auto name = "fneq.write." + std::to_string(bits);
  auto *fn = M.getFunction(name);
  if (!fn) {
    auto &C = M.getContext();
    fn = Function::Create(
        FunctionType::get(Type::getVoidTy(C),
                          {addr_ty, IntegerType::get(C, bits)}, false),
        GlobalValue::ExternalLinkage, name, M);
    fn->setMemoryEffects(MemoryEffects::inaccessibleMemOnly(ModRefInfo::ModRef));
    fn->setDoesNotThrow();
    fn->setWillReturn();
  }
  return fn;
}

// Give a remill intrinsic declaration a body built by `build`.
void DefineIntrinsic(Function *fn,
                     const std::function<void(IRBuilder<> &, Function *)> &build) {
  if (!fn || !fn->isDeclaration() || fn->use_empty()) return;
  fn->removeFnAttr(Attribute::NoInline);
  fn->removeFnAttr(Attribute::OptimizeNone);
  fn->removeFnAttr(Attribute::NoDuplicate);
  fn->addFnAttr(Attribute::AlwaysInline);
  fn->setLinkage(GlobalValue::InternalLinkage);
  auto *bb = BasicBlock::Create(fn->getContext(), "", fn);
  IRBuilder<> ir(bb);
  build(ir, fn);
}

Value *Coerce(IRBuilder<> &ir, Value *v, Type *to) {
  Type *from = v->getType();
  if (from == to) return v;
  if (!from->isSized() || !to->isSized())
    Fail("cannot coerce between unsized types");
  auto &DL = ir.GetInsertBlock()->getModule()->getDataLayout();
  bool same_size = DL.getTypeSizeInBits(from) == DL.getTypeSizeInBits(to);
  bool both_int = from->isIntegerTy() && to->isIntegerTy();
  if (from->isAggregateType() || to->isAggregateType() || (!same_size && !both_int)) {
    // bitcast cannot produce or consume aggregates or change sizes (e.g. an
    // 80-bit x87 value in a 16-byte slot): go through a zeroed stack slot
    // (SROA turns this back into SSA values).
    Function *fn = ir.GetInsertBlock()->getParent();
    BasicBlock &entry = fn->getEntryBlock();
    IRBuilder<> eb(&entry, entry.getFirstInsertionPt());  // valid even if empty
    uint64_t sz = std::max(DL.getTypeAllocSize(from), DL.getTypeAllocSize(to));
    auto *slot = eb.CreateAlloca(ArrayType::get(eb.getInt8Ty(), sz));
    ir.CreateMemSet(slot, ir.getInt8(0), sz, MaybeAlign(1));
    ir.CreateStore(v, slot);
    return ir.CreateLoad(to, slot);
  }
  if (DL.getTypeSizeInBits(from) == DL.getTypeSizeInBits(to))
    return ir.CreateBitCast(v, to);
  if (from->isIntegerTy() && to->isIntegerTy())
    return ir.CreateZExtOrTrunc(v, to);
  Fail("cannot coerce value between incompatible types");
}

// ------------------------------------------------------- UB-removal utilities

// Drop every flag, attribute and metadata item whose violation would yield
// poison or UB. None of them is justified by machine semantics.
void StripPoison(Function &F) {
  for (auto &arg : F.args()) {
    arg.removeAttr(Attribute::NoUndef);
    arg.removeAttr(Attribute::NoAlias);
    arg.removeAttr(Attribute::NonNull);
    arg.removeAttr(Attribute::Range);
  }
  F.removeRetAttr(Attribute::NoUndef);
  F.removeRetAttr(Attribute::Range);
  std::vector<Instruction *> assumes;
  for (auto &I : instructions(F)) {
    I.dropPoisonGeneratingFlags();
    I.dropPoisonGeneratingMetadata();
    I.setMetadata(LLVMContext::MD_tbaa, nullptr);
    I.setMetadata(LLVMContext::MD_alias_scope, nullptr);
    I.setMetadata(LLVMContext::MD_noalias, nullptr);
    I.setMetadata(LLVMContext::MD_noundef, nullptr);
    if (auto *fp = dyn_cast<FPMathOperator>(&I)) {
      (void) fp;
      I.copyFastMathFlags(FastMathFlags());
    }
    if (auto *cb = dyn_cast<CallBase>(&I)) {
      for (unsigned i = 0; i < cb->arg_size(); ++i) {
        cb->removeParamAttr(i, Attribute::NoUndef);
        cb->removeParamAttr(i, Attribute::NonNull);
        cb->removeParamAttr(i, Attribute::Range);
      }
      cb->removeRetAttr(Attribute::NoUndef);
      cb->removeRetAttr(Attribute::Range);
      cb->removeRetAttr(Attribute::NonNull);
      if (auto *ii = dyn_cast<IntrinsicInst>(cb)) {
        auto id = ii->getIntrinsicID();
        if (id == Intrinsic::assume) assumes.push_back(ii);
        // ctlz/cttz(x, is_zero_poison=true) -> false
        if ((id == Intrinsic::ctlz || id == Intrinsic::cttz) &&
            !cast<ConstantInt>(ii->getArgOperand(1))->isZero())
          ii->setArgOperand(1, ConstantInt::getFalse(F.getContext()));
        if (id == Intrinsic::abs &&
            !cast<ConstantInt>(ii->getArgOperand(1))->isZero())
          ii->setArgOperand(1, ConstantInt::getFalse(F.getContext()));
      }
    }
  }
  for (auto *a : assumes) a->eraseFromParent();
}

// Shifts by >= bit width are poison in LLVM. Replace them with a total
// definition: the shift is computed on the masked amount and the
// out-of-range case selects the mathematically saturated value.
void MakeShiftsTotal(Function &F) {
  std::vector<BinaryOperator *> shifts;
  for (auto &I : instructions(F))
    if (auto *bo = dyn_cast<BinaryOperator>(&I))
      if (bo->isShift() && bo->getType()->isIntOrIntVectorTy()) shifts.push_back(bo);
  for (auto *bo : shifts) {
    auto *ty = bo->getType();
    unsigned w = ty->getScalarSizeInBits();
    auto *amt = bo->getOperand(1);
    if (auto *c = dyn_cast<ConstantInt>(amt))
      if (c->getValue().ult(w)) continue;
    IRBuilder<> ir(bo);
    auto *wc = ConstantInt::get(ty, w);
    auto *in_range = ir.CreateICmpULT(amt, wc);
    auto *masked = ir.CreateAnd(amt, ConstantInt::get(ty, w - 1));
    auto *shifted = ir.CreateBinOp(bo->getOpcode(), bo->getOperand(0), masked);
    Value *sat = bo->getOpcode() == Instruction::AShr
                     ? ir.CreateAShr(bo->getOperand(0), ConstantInt::get(ty, w - 1))
                     : Constant::getNullValue(ty);
    auto *res = ir.CreateSelect(in_range, shifted, sat);
    bo->replaceAllUsesWith(res);
    bo->eraseFromParent();
  }
}

// Integer division by zero (and signed overflow) is UB in LLVM but a #DE
// fault on x86. Branch to the fault exit before the division can fault, so the
// division itself only ever executes with valid operands.
void BranchToFault(Value *cond, Instruction *before, BasicBlock *fault_bb) {
  Instruction *then = SplitBlockAndInsertIfThen(cond, before->getIterator(), true);
  IRBuilder<> tb(then);
  tb.CreateBr(fault_bb);
  then->eraseFromParent();
}

void GuardDivisions(Function &F, BasicBlock *fault_bb) {
  std::vector<BinaryOperator *> divs;
  for (auto &I : instructions(F))
    if (auto *bo = dyn_cast<BinaryOperator>(&I)) switch (bo->getOpcode()) {
        case Instruction::UDiv: case Instruction::SDiv:
        case Instruction::URem: case Instruction::SRem:
          divs.push_back(bo);
          break;
        default: break;
      }
  for (auto *bo : divs) {
    if (bo->getType()->isVectorTy()) {
      Unsupported("vector integer division");
      continue;
    }
    IRBuilder<> ir(bo);
    auto *ty = bo->getType();
    auto *a = bo->getOperand(0), *b = bo->getOperand(1);
    Value *bad = ir.CreateICmpEQ(b, Constant::getNullValue(ty));
    bool is_signed = bo->getOpcode() == Instruction::SDiv ||
                     bo->getOpcode() == Instruction::SRem;
    if (is_signed) {
      auto *ovf = ir.CreateAnd(
          ir.CreateICmpEQ(a, ConstantInt::get(ty, APInt::getSignedMinValue(ty->getIntegerBitWidth()))),
          ir.CreateICmpEQ(b, Constant::getAllOnesValue(ty)));
      bad = ir.CreateOr(bad, ovf);
    }
    BranchToFault(bad, bo, fault_bb);
  }
}

// Before optimisation the lowered function must be total and deterministic:
// no undef/poison values anywhere. Any left in remill's semantics are replaced
// by zero (and reported). Every later LLVM pass is a refinement, and a
// refinement of a total deterministic function computes the same outputs, so
// poison that the optimiser introduces (e.g. in unused vector lanes) can never
// reach an observable result.
unsigned MakeTotal(Function &F) {
  unsigned n = 0;
  for (auto &I : instructions(F))
    for (auto &u : I.operands())
      if (isa<UndefValue>(u.get()) && !isa<BasicBlock>(u.get())) {
        u.set(Constant::getNullValue(u->getType()));
        ++n;
      }
  return n;
}

// Final audit: nothing that lets Alive2 treat some input as undefined, other
// than the fault exit (added later by --finalize).
void Audit(Function &F) {
  for (auto &I : instructions(F)) {
    if (isa<UnreachableInst>(&I)) Fail("audit: unreachable in lowered IR");
    if (auto *cb = dyn_cast<CallBase>(&I))
      if (auto *ii = dyn_cast<IntrinsicInst>(cb))
        if (ii->getIntrinsicID() == Intrinsic::assume)
          Fail("audit: llvm.assume in lowered IR");
    if (auto *op = dyn_cast<Operator>(&I)) {
      if (op->hasPoisonGeneratingFlags())
        Fail(std::string("audit: poison-generating flag survived: ") + I.getOpcodeName());
    }
  }
}

// remill computes the signed-overflow flag arithmetically:
//   sub: OF = 2 == (sign(a) ^ sign(b)) + (sign(a) ^ sign(a - b))
//   add: OF = 2 == (sign(a) ^ sign(r)) + (sign(b) ^ sign(r)),  r = a + b
// The sum is 2 exactly when the sign bits indicate signed overflow, so this is
// the same value as llvm.s{sub,add}.with.overflow. InstCombine folds conditions
// built from the intrinsic (e.g. jle: ZF | SF != OF  ->  icmp sle a, b), which in
// turn lets SCEV compute loop trip counts. Pure canonicalisation: same values.
unsigned CanonicalizeOverflowFlags(Function &F) {
  using namespace llvm::PatternMatch;
  std::vector<ICmpInst *> cands;
  for (auto &I : instructions(F))
    if (auto *ic = dyn_cast<ICmpInst>(&I))
      if (ic->getPredicate() == ICmpInst::ICMP_EQ) cands.push_back(ic);
  unsigned n = 0;
  for (auto *ic : cands) {
    const APInt *two;
    Value *P, *Q;
    if (!match(ic->getOperand(1), m_APInt(two)) || *two != 2) continue;
    if (!match(ic->getOperand(0), m_Add(m_Value(P), m_Value(Q)))) continue;
    // sign(v) is lshr(v, W-1)
    auto sign_of = [](Value *v, Value *&inner) {
      const APInt *sh;
      if (!match(v, m_LShr(m_Value(inner), m_APInt(sh)))) return false;
      return inner->getType()->isIntegerTy() &&
             sh->getZExtValue() == inner->getType()->getIntegerBitWidth() - 1;
    };
    Value *x, *y;
    if (!sign_of(P, x) || !sign_of(Q, y)) continue;
    Value *A = nullptr, *B = nullptr, *R = nullptr;
    Intrinsic::ID id = Intrinsic::not_intrinsic;
    for (int swap = 0; swap < 2 && !A; ++swap) {
      Value *t1 = swap ? y : x, *t2 = swap ? x : y;
      Value *o0, *o1;
      if (!match(t1, m_Xor(m_Value(o0), m_Value(o1)))) continue;
      for (int k = 0; k < 2 && !A; ++k) {
        Value *a = k ? o1 : o0, *r = k ? o0 : o1;
        Value *b;
        const APInt *c;
        // SUB: r = a - b (or a + (-C)); t2 = a ^ b, or a when b >= 0 is constant.
        bool is_sub = false;
        if (match(r, m_Sub(m_Specific(a), m_Value(b)))) {
          is_sub = true;
        } else if (match(r, m_Add(m_Specific(a), m_APInt(c)))) {
          b = ConstantInt::get(a->getType(), -*c);
          is_sub = true;
        }
        if (is_sub) {
          auto *cb = dyn_cast<ConstantInt>(b);
          if (match(t2, m_c_Xor(m_Specific(a), m_Specific(b))) ||
              (cb && !cb->isNegative() && t2 == a)) {
            A = a; B = b; R = r; id = Intrinsic::ssub_with_overflow;
            break;
          }
        }
        // ADD: r = a + b; t2 = b ^ r.
        Value *b2;
        if (match(r, m_c_Add(m_Specific(a), m_Value(b2))) &&
            match(t2, m_c_Xor(m_Specific(b2), m_Specific(r)))) {
          A = a; B = b2; R = r; id = Intrinsic::sadd_with_overflow;
        }
      }
    }
    if (!A) continue;
    // Compute the intrinsic right after the arithmetic result R (which dominates
    // all of R's users, and is dominated by A and B), and let R's users (e.g. the
    // sign flag) read the intrinsic's result field: the same value, but now
    // InstCombine sees one operation and can fold the flag logic.
    auto *RI = dyn_cast<Instruction>(R);
    IRBuilder<> ir(RI ? RI->getNextNode() : ic);
    if (RI && isa<PHINode>(RI)) ir.SetInsertPoint(RI->getParent()->getFirstNonPHIIt());
    Value *res = ir.CreateBinaryIntrinsic(id, A, B);
    Value *val = ir.CreateExtractValue(res, 0);
    Value *ovf = ir.CreateExtractValue(res, 1);
    ic->replaceAllUsesWith(ovf);
    ic->eraseFromParent();
    if (RI) RI->replaceUsesWithIf(val, [&](Use &u) { return u.getUser() != val && u.getUser() != res; });
    ++n;
  }

  // SF != OF, i.e. xor(ovf, (res <s 0)), is the signed "less than" of the
  // operation's operands:  ssub(a, b) -> a <s b;  sadd(a, C) -> a <s -C (C != MIN).
  std::vector<BinaryOperator *> xors;
  for (auto &I : instructions(F))
    if (auto *bo = dyn_cast<BinaryOperator>(&I))
      if (bo->getOpcode() == Instruction::Xor && bo->getType()->isIntegerTy(1)) xors.push_back(bo);
  for (auto *x : xors) {
    for (int k = 0; k < 2; ++k) {
      Value *ovf_v = x->getOperand(k), *sf_v = x->getOperand(1 - k);
      auto *ev1 = dyn_cast<ExtractValueInst>(ovf_v);
      if (!ev1 || ev1->getIndices()[0] != 1) continue;
      auto *call = dyn_cast<IntrinsicInst>(ev1->getAggregateOperand());
      if (!call) continue;
      Value *res0 = nullptr;
      CmpPredicate pred;
      const APInt *zero;
      if (!match(sf_v, m_ICmp(pred, m_Value(res0), m_APInt(zero))) || pred != ICmpInst::ICMP_SLT ||
          !zero->isZero())
        continue;
      auto *ev0 = dyn_cast<ExtractValueInst>(res0);
      if (!ev0 || ev0->getIndices()[0] != 0 || ev0->getAggregateOperand() != call) continue;
      Value *a = call->getArgOperand(0), *b = call->getArgOperand(1);
      Value *lt = nullptr;
      IRBuilder<> ir(x);
      if (call->getIntrinsicID() == Intrinsic::ssub_with_overflow) {
        lt = ir.CreateICmpSLT(a, b);
      } else if (call->getIntrinsicID() == Intrinsic::sadd_with_overflow) {
        auto *c = dyn_cast<ConstantInt>(b);
        if (!c || c->getValue().isMinSignedValue()) continue;
        lt = ir.CreateICmpSLT(a, ConstantInt::get(c->getType(), -c->getValue()));
      }
      if (!lt) continue;
      x->replaceAllUsesWith(lt);
      x->eraseFromParent();
      ++n;
      break;
    }
  }
  return n;
}

// Alive2 does not support x86_fp80. Widening a float/double to the 80-bit x87
// format (FLD m32/m64) is exact, so it is rewritten as integer operations on the
// IEEE encoding; the bitcasts around it then fold away. Real x87 arithmetic on
// fp80 values is left alone (Alive2 then reports it as unsupported).
unsigned LowerFPExtToFP80(Function &F) {
  std::vector<FPExtInst *> exts;
  for (auto &I : instructions(F))
    if (auto *fe = dyn_cast<FPExtInst>(&I))
      if (fe->getType()->isX86_FP80Ty() &&
          (fe->getSrcTy()->isFloatTy() || fe->getSrcTy()->isDoubleTy()))
        exts.push_back(fe);
  for (auto *fe : exts) {
    IRBuilder<> b(fe);
    bool dbl = fe->getSrcTy()->isDoubleTy();
    unsigned W = dbl ? 64 : 32, EB = dbl ? 11 : 8, FB = dbl ? 52 : 23;
    int64_t bias_delta = 16383 - (dbl ? 1023 : 127);
    auto *iw = b.getIntNTy(W);
    auto *i80 = b.getIntNTy(80);
    auto *i64t = b.getInt64Ty();
    Value *bits = b.CreateBitCast(fe->getOperand(0), iw);
    Value *sign = b.CreateLShr(bits, W - 1);
    Value *exp = b.CreateAnd(b.CreateLShr(bits, FB), (1ULL << EB) - 1);
    Value *frac = b.CreateAnd(bits, (1ULL << FB) - 1);
    Value *frac64 = b.CreateShl(b.CreateZExt(frac, i64t), 63 - FB);  // fraction just below bit 63
    Value *one = ConstantInt::get(i64t, 1ULL << 63);
    // normal: implicit 1 made explicit, exponent rebased
    Value *n_mant = b.CreateOr(one, frac64);
    Value *n_exp = b.CreateAdd(b.CreateZExt(exp, i64t), ConstantInt::get(i64t, bias_delta));
    // denormal: normalise (frac != 0); value = frac64/2^63 * 2^(1-bias)
    Function *ctlz = Intrinsic::getOrInsertDeclaration(F.getParent(), Intrinsic::ctlz, {i64t});
    Value *lz = b.CreateCall(ctlz, {frac64, b.getFalse()});
    Value *d_mant = b.CreateShl(frac64, lz);
    Value *d_exp = b.CreateSub(ConstantInt::get(i64t, bias_delta + 1), lz);
    // inf / nan: maximal exponent; a signalling NaN is quietened (as FLD does)
    Value *quiet = ConstantInt::get(i64t, 1ULL << 62);
    Value *is_nan = b.CreateICmpNE(frac, ConstantInt::get(iw, 0));
    Value *s_mant = b.CreateSelect(is_nan, b.CreateOr(n_mant, quiet), n_mant);
    Value *e_max = ConstantInt::get(iw, (1ULL << EB) - 1);
    Value *is_special = b.CreateICmpEQ(exp, e_max);
    Value *is_zero_exp = b.CreateICmpEQ(exp, ConstantInt::get(iw, 0));
    Value *is_zero = b.CreateAnd(is_zero_exp, b.CreateICmpEQ(frac, ConstantInt::get(iw, 0)));
    Value *mant = b.CreateSelect(is_special, s_mant, b.CreateSelect(is_zero_exp, d_mant, n_mant));
    Value *e80 = b.CreateSelect(is_special, ConstantInt::get(i64t, 0x7FFF),
                                b.CreateSelect(is_zero_exp, d_exp, n_exp));
    mant = b.CreateSelect(is_zero, ConstantInt::get(i64t, 0), mant);
    e80 = b.CreateSelect(is_zero, ConstantInt::get(i64t, 0), e80);
    Value *res = b.CreateOr(b.CreateShl(b.CreateZExt(sign, i80), 79),
                            b.CreateOr(b.CreateShl(b.CreateZExt(e80, i80), 64),
                                       b.CreateZExt(mant, i80)));
    fe->replaceAllUsesWith(b.CreateBitCast(res, fe->getType()));
    fe->eraseFromParent();
  }
  return static_cast<unsigned>(exts.size());
}

// A wide unsigned division whose operands provably fit in fewer bits computes
// the same value as the narrow division, zero-extended. remill's 64-bit DIV
// semantics on zero-extended 32-bit operands are much harder for the solver than
// the 32-bit DIV the other side may use; this makes both sides look alike.
unsigned NarrowUnsignedDivisions(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  std::vector<BinaryOperator *> divs;
  for (auto &I : instructions(F))
    if (auto *bo = dyn_cast<BinaryOperator>(&I))
      if ((bo->getOpcode() == Instruction::UDiv || bo->getOpcode() == Instruction::URem) &&
          bo->getType()->isIntegerTy())
        divs.push_back(bo);
  unsigned n = 0;
  for (auto *bo : divs) {
    unsigned W = bo->getType()->getIntegerBitWidth();
    KnownBits ka = computeKnownBits(bo->getOperand(0), DL);
    KnownBits kb = computeKnownBits(bo->getOperand(1), DL);
    unsigned need = std::max(ka.countMaxActiveBits(), kb.countMaxActiveBits());
    unsigned narrow = need <= 8 ? 8 : need <= 16 ? 16 : need <= 32 ? 32 : W;
    if (narrow >= W) continue;
    IRBuilder<> ir(bo);
    auto *nt = ir.getIntNTy(narrow);
    Value *a = ir.CreateTrunc(bo->getOperand(0), nt);
    Value *b = ir.CreateTrunc(bo->getOperand(1), nt);
    Value *r = ir.CreateBinOp(bo->getOpcode(), a, b);
    bo->replaceAllUsesWith(ir.CreateZExt(r, bo->getType()));
    bo->eraseFromParent();
    ++n;
  }
  return n;
}

// ---------------------------------------------- stack-provenance utilities

bool IsCallTo(Value *v, Function *fn) {
  auto *cb = dyn_cast<CallBase>(v);
  return cb && cb->getCalledFunction() == fn;
}

// Does `v` (transitively, through pure integer ops) depend on a call to the
// marker function `marker`? The marker is pure, so passes may delete, merge
// or duplicate its calls; we therefore never hold on to a particular call.
bool DependsOn(Value *v, Function *marker, std::set<Value *> &seen) {
  if (IsCallTo(v, marker)) return true;
  if (!seen.insert(v).second) return false;
  auto *I = dyn_cast<Instruction>(v);
  if (!I) return false;
  if (isa<CallBase>(I) || isa<LoadInst>(I)) return false;
  for (auto &op : I->operands())
    if (DependsOn(op.get(), marker, seen)) return true;
  return false;
}

bool DependsOn(Value *v, Function *marker) {
  std::set<Value *> seen;
  return DependsOn(v, marker, seen);
}

// Does `v` depend on the result of a not-yet-lowered memory read?
bool DependsOnPendingRead(Value *v, std::set<Value *> &seen) {
  if (!seen.insert(v).second) return false;
  auto *I = dyn_cast<Instruction>(v);
  if (!I) return false;
  if (auto *cb = dyn_cast<CallBase>(I))
    if (auto *fn = cb->getCalledFunction())
      if (fn->getName().starts_with("fneq.read.")) return true;
  for (auto &op : I->operands())
    if (DependsOnPendingRead(op.get(), seen)) return true;
  return false;
}

}  // namespace

// ======================================================================= main

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  if (!FLAGS_finalize.empty()) {
    // After the last optimisation: the fault marker becomes UB (see the comment
    // on fault_exit). changeToUnreachable also fixes up successor phis.
    if (FLAGS_out.empty()) Fail("--finalize needs --out");
    LLVMContext C;
    SMDiagnostic err;
    auto M = parseIRFile(FLAGS_finalize, err, C);
    if (!M) Fail("cannot parse " + FLAGS_finalize + ": " + err.getMessage().str());
    std::vector<CallBase *> marks;
    if (auto *f = M->getFunction("fneq.fault"))
      for (auto *u : f->users())
        if (auto *cb = dyn_cast<CallBase>(u)) marks.push_back(cb);
    for (auto *cb : marks) changeToUnreachable(cb);
    if (auto *f = M->getFunction("fneq.fault"))
      if (f->use_empty()) f->eraseFromParent();
    std::string why;
    raw_string_ostream vs(why);
    if (verifyModule(*M, &vs)) Fail("finalized module does not verify: " + why.substr(0, 600));
    std::error_code ec;
    raw_fd_ostream os(FLAGS_out, ec);
    if (ec) Fail("cannot write " + FLAGS_out);
    M->print(os, nullptr);
    os.close();
    g_notes.push_back("fault markers rewritten: " + std::to_string(marks.size()));
    Finish(true);
  }
  if (FLAGS_spec.empty() || FLAGS_out.empty()) Fail("--spec and --out are required");
  const Spec spec = LoadSpec(FLAGS_spec);

  LLVMContext C;
  auto arch = remill::Arch::Get(C, spec.os, spec.arch);
  if (!arch) Fail("unsupported arch/os " + spec.arch + "/" + spec.os);
  std::unique_ptr<Module> M(remill::LoadArchSemantics(arch.get()));
  if (!M) Fail("could not load remill semantics");
  remill::IntrinsicTable intrinsics(M.get());
  auto inst_lifter = arch->DefaultLifter(intrinsics);
  (void) inst_lifter;

  const unsigned abits = arch->address_size;
  auto *addr_ty = IntegerType::get(C, abits);
  const unsigned abytes = abits / 8;

  // ---------------------------------------------------------------- 1. lift
  Manager manager(arch.get(), M.get(), spec);
  remill::TraceLifter trace_lifter(arch.get(), manager);
  if (!trace_lifter.Lift(spec.entry)) Fail("remill TraceLifter failed");
  auto entry_it = manager.traces.find(spec.entry);
  if (entry_it == manager.traces.end()) Fail("entry trace was not lifted");
  Function *entry_fn = entry_it->second;

  if (auto *g = M->getGlobalVariable("llvm.compiler.used", true)) g->eraseFromParent();
  {
    std::vector<GlobalVariable *> dead;
    for (auto &g : M->globals())
      if (g.getName().starts_with("ISEL_")) dead.push_back(&g);
    for (auto *g : dead) g->eraseFromParent();
    if (auto *f = M->getFunction("__remill_intrinsics")) f->eraseFromParent();
  }
  if (!FLAGS_pre_out.empty()) {
    std::error_code ec;
    raw_fd_ostream os(FLAGS_pre_out, ec);
    for (auto &kv : manager.traces) kv.second->print(os);
  }

  // Control-flow intrinsics that we cannot model make the result UNKNOWN.
  // Record them now, with the lifted PC when it is a constant.
  auto record_uses = [&](Function *fn, const std::string &what) {
    if (!fn) return;
    for (auto *u : fn->users())
      if (auto *cb = dyn_cast<CallBase>(u)) {
        bool in_trace = false;
        for (auto &kv : manager.traces) in_trace |= cb->getFunction() == kv.second;
        if (!in_trace) continue;
        std::string where;
        if (auto *pc = dyn_cast<ConstantInt>(cb->getArgOperand(remill::kPCArgNum)))
          where = " at 0x" + utohexstr(pc->getZExtValue());
        Unsupported(what + where);
      }
  };
  record_uses(intrinsics.error, "undecodable or unsupported instruction (__remill_error)");
  record_uses(intrinsics.missing_block, "control flow leaves the known function bytes (__remill_missing_block)");
  record_uses(intrinsics.jump, "unresolved indirect jump (__remill_jump)");
  if (!spec.has_indirect)
    record_uses(intrinsics.function_call, "indirect call (__remill_function_call)");
  record_uses(intrinsics.async_hyper_call, "hyper call / interrupt (__remill_async_hyper_call)");
  if (g_unsupported.size()) Finish(false, "lifting produced unsupported constructs");

  // ------------------------------------------------- 2. define intrinsics
  auto replace_with_arg0 = [&](Function *fn) {
    if (!fn) return;
    std::vector<CallBase *> calls;
    for (auto *u : fn->users())
      if (auto *cb = dyn_cast<CallBase>(u)) calls.push_back(cb);
    for (auto *cb : calls) {
      cb->replaceAllUsesWith(cb->getArgOperand(0));
      cb->eraseFromParent();
    }
  };
  // __remill_flag_computation_* and __remill_compare_* only annotate a value
  // that is already computed: they return their first argument.
  {
    std::vector<Function *> annot;
    for (auto &F : *M)
      if (F.getName().starts_with("__remill_flag_computation_") ||
          F.getName().starts_with("__remill_compare_"))
        annot.push_back(&F);
    for (auto *fn : annot) replace_with_arg0(fn);
  }

  auto define_read = [&](Function *fn, unsigned bits) {
    DefineIntrinsic(fn, [&](IRBuilder<> &ir, Function *f) {
      auto *addr = f->getArg(1);
      Value *v = ir.CreateCall(MemRead(*M, bits, addr_ty),
                               {ir.CreateZExtOrTrunc(addr, addr_ty)});
      ir.CreateRet(Coerce(ir, v, f->getReturnType()));
    });
  };
  auto define_write = [&](Function *fn, unsigned bits) {
    DefineIntrinsic(fn, [&](IRBuilder<> &ir, Function *f) {
      auto *addr = f->getArg(1);
      auto *val = f->getArg(2);
      ir.CreateCall(MemWrite(*M, bits, addr_ty),
                    {ir.CreateZExtOrTrunc(addr, addr_ty),
                     Coerce(ir, val, IntegerType::get(C, bits))});
      ir.CreateRet(f->getArg(0));
    });
  };
  define_read(intrinsics.read_memory_8, 8);
  define_read(intrinsics.read_memory_16, 16);
  define_read(intrinsics.read_memory_32, 32);
  define_read(intrinsics.read_memory_64, 64);
  define_read(intrinsics.read_memory_f32, 32);
  define_read(intrinsics.read_memory_f64, 64);
  define_write(intrinsics.write_memory_8, 8);
  define_write(intrinsics.write_memory_16, 16);
  define_write(intrinsics.write_memory_32, 32);
  define_write(intrinsics.write_memory_64, 64);
  define_write(intrinsics.write_memory_f32, 32);
  define_write(intrinsics.write_memory_f64, 64);

  // f80 accessors take/return the value through memory or by value depending
  // on the remill version; handle the by-value form and flag the rest.
  for (auto *fn : {intrinsics.read_memory_f128, intrinsics.write_memory_f128})
    if (fn && !fn->use_empty())
      Unsupported("128-bit float memory access (remill models it as double)");
  for (auto *fn : {intrinsics.read_memory_f80, intrinsics.write_memory_f80}) {
    if (!fn || fn->use_empty()) continue;
    bool is_read = fn == intrinsics.read_memory_f80;
    unsigned bits = 80;
    auto *FT = fn->getFunctionType();
    if (is_read && FT->getNumParams() == 2 && !FT->getReturnType()->isPointerTy()) {
      define_read(fn, bits);
    } else if (!is_read && FT->getNumParams() == 3 && !FT->getParamType(2)->isPointerTy()) {
      define_write(fn, bits);
    } else if (is_read && FT->getNumParams() == 3 && FT->getParamType(2)->isPointerTy()) {
      // ptr __remill_read_memory_f80(ptr mem, addr, ptr out)
      DefineIntrinsic(fn, [&](IRBuilder<> &ir, Function *f) {
        Value *v = ir.CreateCall(MemRead(*M, bits, addr_ty),
                                 {ir.CreateZExtOrTrunc(f->getArg(1), addr_ty)});
        ir.CreateStore(v, f->getArg(2));
        ir.CreateRet(f->getArg(0));
      });
    } else if (!is_read && FT->getNumParams() == 3 && FT->getParamType(2)->isPointerTy()) {
      DefineIntrinsic(fn, [&](IRBuilder<> &ir, Function *f) {
        Value *v = ir.CreateLoad(IntegerType::get(C, bits), f->getArg(2));
        ir.CreateCall(MemWrite(*M, bits, addr_ty),
                      {ir.CreateZExtOrTrunc(f->getArg(1), addr_ty), v});
        ir.CreateRet(f->getArg(0));
      });
    } else {
      Unsupported("unexpected signature for " + fn->getName().str());
    }
  }

  // Undefined values (e.g. flags after some instructions) are modelled as an
  // arbitrary but fixed value: a pure function with no arguments. Both sides
  // share it, so a function whose output depends on it is only equivalent to
  // one that produces the same unknown value.
  for (auto *fn : {intrinsics.undefined_8, intrinsics.undefined_16, intrinsics.undefined_32,
                   intrinsics.undefined_64, intrinsics.undefined_f32, intrinsics.undefined_f64,
                   intrinsics.undefined_f80}) {
    DefineIntrinsic(fn, [&](IRBuilder<> &ir, Function *f) {
      auto *rt = f->getReturnType();
      if (rt->isVoidTy()) {
        // sret form: void __remill_undefined_f80(ptr sret(float80_t))
        if (f->arg_size() != 1 || !f->hasParamAttribute(0, Attribute::StructRet))
          Fail("unexpected signature for " + f->getName().str());
        auto *u = DeclarePure(*M, "fneq.undef.i80", IntegerType::get(C, 80), {});
        ir.CreateStore(ir.CreateCall(u), f->getArg(0));
        ir.CreateRetVoid();
        return;
      }
      auto bits = M->getDataLayout().getTypeSizeInBits(rt).getFixedValue();
      auto *ity = IntegerType::get(C, bits);
      auto *u = DeclarePure(*M, "fneq.undef.i" + std::to_string(bits), ity, {});
      ir.CreateRet(Coerce(ir, ir.CreateCall(u), rt));
    });
  }
  for (auto *fn : {intrinsics.barrier_load_load, intrinsics.barrier_load_store,
                   intrinsics.barrier_store_load, intrinsics.barrier_store_store,
                   intrinsics.atomic_begin, intrinsics.atomic_end,
                   intrinsics.delay_slot_begin, intrinsics.delay_slot_end}) {
    DefineIntrinsic(fn, [&](IRBuilder<> &ir, Function *f) {
      ir.CreateRet(f->getArg(0));
    });
  }
  DefineIntrinsic(intrinsics.function_return, [&](IRBuilder<> &ir, Function *f) {
    ir.CreateRet(f->getArg(remill::kMemoryPointerArgNum));
  });

  // ------------------------------------------------------ 3. callee stubs
  const remill::Register *sp_reg = arch->RegisterByName(arch->StackPointerRegisterName());
  const remill::Register *pc_reg = arch->RegisterByName(arch->ProgramCounterRegisterName());
  if (!sp_reg || !pc_reg) Fail("no SP/PC register");

  auto reg_by_name = [&](const std::string &name) -> const remill::Register * {
    auto *r = arch->RegisterByName(name);
    if (!r) Fail("unknown register " + name);
    return r;
  };
  auto reg_int_ty = [&](const remill::Register *r) {
    return IntegerType::get(C, r->size * 8);
  };
  // "NAME" or "NAME:BITS" (the low BITS bits of the register).
  struct RegRef { const remill::Register *reg; unsigned bits; };
  auto reg_ref = [&](const std::string &name) -> RegRef {
    auto colon = name.find(':');
    auto *r = reg_by_name(name.substr(0, colon));
    unsigned bits = r->size * 8;
    if (colon != std::string::npos) bits = std::stoul(name.substr(colon + 1));
    if (bits == 0 || bits > r->size * 8) Fail("bad register width in " + name);
    return {r, bits};
  };
  auto load_reg = [&](IRBuilder<> &b, Value *st, const RegRef &rr) -> Value * {
    Value *v = Coerce(b, b.CreateLoad(rr.reg->type, rr.reg->AddressOf(st, b)), reg_int_ty(rr.reg));
    return rr.bits < rr.reg->size * 8 ? b.CreateTrunc(v, IntegerType::get(C, rr.bits)) : v;
  };
  // Values wider than 64 bits cross call/return boundaries as 64-bit chunks (low
  // first): Alive2 fails to type-check structs with padded i128 members.
  auto chunks = [](unsigned bits) {
    std::vector<unsigned> out;
    for (unsigned done = 0; done < bits; done += 64) out.push_back(std::min(64u, bits - done));
    return out;
  };

  // Translate a pointer-sized value through spec.value_remap (identity elsewhere).
  auto remap_value = [&](IRBuilder<> &b, Value *v) -> Value * {
    if (spec.value_remap.empty() || v->getType() != addr_ty) return v;
    Value *out = v;
    for (auto &r : spec.value_remap) {
      Value *rel = b.CreateSub(v, ConstantInt::get(addr_ty, r.from));
      Value *in = b.CreateICmpULT(rel, ConstantInt::get(addr_ty, r.size));
      out = b.CreateSelect(in, b.CreateAdd(rel, ConstantInt::get(addr_ty, r.to)), out);
    }
    return out;
  };
  // A callee stub: call @ext.<name>(args...), store its results and clobbers,
  // then return like the callee's `ret`. For an indirect call the target value
  // (the stub's PC argument) is passed as an extra first argument.
  auto define_stub = [&](Function *decl, const Callee &cs, bool indirect) {
    std::vector<Type *> arg_tys;
    if (indirect) arg_tys.push_back(addr_ty);
    for (auto &a : cs.args)
      arg_tys.push_back(a.reg.empty() ? (Type *) IntegerType::get(C, a.size * 8)
                                      : (Type *) IntegerType::get(C, reg_ref(a.reg).bits));
    // Results wider than 64 bits (XMM registers) travel as 64-bit chunks.
    std::vector<Type *> res_tys;
    for (auto &r : cs.results)
      for (unsigned w : chunks(reg_by_name(r)->size * 8)) res_tys.push_back(IntegerType::get(C, w));
    auto *res_ty = StructType::get(C, res_tys);
    auto *ext = Function::Create(FunctionType::get(res_ty, arg_tys, false),
                                 GlobalValue::ExternalLinkage, "ext." + cs.name, *M);
    ext->setDoesNotThrow();
    ext->setWillReturn();

    decl->setLinkage(GlobalValue::ExternalLinkage);
    DefineIntrinsic(decl, [&](IRBuilder<> &ir, Function *f) {
      auto *state = f->getArg(remill::kStatePointerArgNum);
      auto *sp_ptr = sp_reg->AddressOf(state, ir);
      Value *sp = ir.CreateLoad(sp_reg->type, sp_ptr);
      // The return address the call pushed; read before the callee runs.
      Value *ra = ir.CreateCall(MemRead(*M, abits, addr_ty), {sp});
      std::vector<Value *> args;
      if (indirect) args.push_back(ir.CreateZExtOrTrunc(f->getArg(remill::kPCArgNum), addr_ty));
      for (auto &a : cs.args) {
        if (a.reg.empty()) {
          auto *addr = ir.CreateAdd(sp, ConstantInt::get(addr_ty, a.offset));
          args.push_back(ir.CreateCall(MemRead(*M, a.size * 8, addr_ty), {addr}));
        } else {
          args.push_back(load_reg(ir, state, reg_ref(a.reg)));
        }
      }
      // frame_visible: markers replaced in step 7 by copies of the frame to and
      // from %M (the callee's view of memory).
      if (spec.frame_visible)
        ir.CreateCall(M->getOrInsertFunction("fneq.frame_out", Type::getVoidTy(C), addr_ty), {sp});
      for (auto &v : args) v = remap_value(ir, v);
      auto *res = ir.CreateCall(ext, args);
      if (spec.frame_visible)
        ir.CreateCall(M->getOrInsertFunction("fneq.frame_back", Type::getVoidTy(C)), {});
      unsigned field = 0;
      for (auto &rn : cs.results) {
        auto *r = reg_by_name(rn);
        auto *wide = reg_int_ty(r);
        Value *v = ConstantInt::get(wide, 0);
        unsigned shift = 0;
        for (unsigned w : chunks(r->size * 8)) {
          Value *part = ir.CreateZExt(ir.CreateExtractValue(res, field++), wide);
          v = ir.CreateOr(v, ir.CreateShl(part, shift));
          shift += w;
        }
        ir.CreateStore(Coerce(ir, v, r->type), r->AddressOf(state, ir));
      }
      // The callee's `ret` pops the return address (and `pop` more bytes);
      // the PC becomes the return address, as after a real `ret`.
      ir.CreateStore(ra, pc_reg->AddressOf(state, ir));
      ir.CreateStore(ir.CreateAdd(sp, ConstantInt::get(addr_ty, abytes + cs.pop)), sp_ptr);
      ir.CreateRet(f->getArg(remill::kMemoryPointerArgNum));
    });
  };
  for (auto &kv : spec.callees) define_stub(manager.External(kv.first), kv.second, false);
  if (spec.has_indirect && intrinsics.function_call && !intrinsics.function_call->use_empty())
    define_stub(intrinsics.function_call, spec.indirect, true);

  // --------------------------------------------------------- 4. wrapper
  std::vector<Type *> param_tys{PointerType::get(C, 0)};
  for (auto &r : spec.in_regs) param_tys.push_back(reg_int_ty(reg_by_name(r)));
  for (auto &a : spec.stack_args) param_tys.push_back(IntegerType::get(C, a.size * 8));
  std::vector<Type *> out_tys;
  // Outputs wider than 64 bits are returned as 64-bit chunks (low first): Alive2
  // fails to type-check returned structs with padded i128 members.
  for (auto &r : spec.out_regs)
    for (unsigned w : chunks(reg_ref(r).bits)) out_tys.push_back(IntegerType::get(C, w));
  out_tys.push_back(addr_ty);               // SP delta
  out_tys.push_back(addr_ty);               // PC after return
  out_tys.push_back(Type::getInt8Ty(C));    // CPU fault
  auto *ret_ty = StructType::get(C, out_tys);
  auto *W = Function::Create(FunctionType::get(ret_ty, param_tys, false),
                             GlobalValue::ExternalLinkage, "fneq_wrapper", *M);
  W->getArg(0)->setName("M");
  {
    unsigned i = 1;
    for (auto &r : spec.in_regs) W->getArg(i++)->setName(StringRef(r).lower());
    for (auto &a : spec.stack_args)
      W->getArg(i++)->setName("stk_" + std::to_string(a.offset));
  }
  auto *wentry = BasicBlock::Create(C, "entry", W);
  IRBuilder<> ir(wentry);
  auto *state = ir.CreateAlloca(arch->StateStructType(), nullptr, "state");
  auto &DL0 = M->getDataLayout();
  ir.CreateMemSet(state, ir.getInt8(0), DL0.getTypeAllocSize(arch->StateStructType()),
                  MaybeAlign(16));
  auto *esp0_fn = DeclarePure(*M, "fneq.esp0", addr_ty, {});
  Value *esp0 = ir.CreateCall(esp0_fn, {}, "esp0");
  ir.CreateStore(esp0, sp_reg->AddressOf(state, ir));
  ir.CreateStore(ConstantInt::get(pc_reg->type, spec.entry), pc_reg->AddressOf(state, ir));
  for (auto &kv : spec.fixed_regs) {
    auto *r = reg_by_name(kv.first);
    ir.CreateStore(Coerce(ir, ConstantInt::get(reg_int_ty(r), kv.second), r->type),
                   r->AddressOf(state, ir));
  }
  {
    unsigned i = 1;
    for (auto &rn : spec.in_regs) {
      auto *r = reg_by_name(rn);
      ir.CreateStore(Coerce(ir, W->getArg(i++), r->type), r->AddressOf(state, ir));
    }
    ir.CreateCall(MemWrite(*M, abits, addr_ty), {esp0, ConstantInt::get(addr_ty, spec.ret_addr)});
    for (auto &a : spec.stack_args) {
      auto *addr = ir.CreateAdd(esp0, ConstantInt::get(addr_ty, a.offset));
      ir.CreateCall(MemWrite(*M, a.size * 8, addr_ty), {addr, W->getArg(i++)});
    }
  }
  Value *mem_token = ConstantPointerNull::get(PointerType::get(C, 0));
  Value *trace_args[remill::kNumBlockArgs];
  trace_args[remill::kStatePointerArgNum] = state;
  trace_args[remill::kPCArgNum] = ConstantInt::get(addr_ty, spec.entry);
  trace_args[remill::kMemoryPointerArgNum] = mem_token;
  ir.CreateCall(entry_fn, trace_args);
  Value *result = ConstantAggregateZero::get(ret_ty);  // fully overwritten below
  unsigned oi = 0;
  for (auto &rn : spec.out_regs) {
    Value *v = load_reg(ir, state, reg_ref(rn));
    if (spec.value_remap_outputs.count(rn)) v = remap_value(ir, v);
    unsigned shift = 0;
    for (unsigned w : chunks(reg_ref(rn).bits)) {
      Value *part = ir.CreateTrunc(ir.CreateLShr(v, shift), IntegerType::get(C, w));
      result = ir.CreateInsertValue(result, part, oi++);
      shift += w;
    }
  }
  Value *sp_final = ir.CreateLoad(sp_reg->type, sp_reg->AddressOf(state, ir));
  result = ir.CreateInsertValue(result, ir.CreateSub(sp_final, esp0), oi++);
  result = ir.CreateInsertValue(result, ir.CreateLoad(pc_reg->type, pc_reg->AddressOf(state, ir)), oi++);
  result = ir.CreateInsertValue(result, ir.getInt8(0), oi++);
  ir.CreateRet(result);

  // A CPU fault ends the function. In the final IR given to Alive2 the fault
  // exit becomes `unreachable` (the only UB left): with refinement checked in
  // both directions, two functions then agree on an input exactly when both
  // fault, or neither faults and all effects are equal. Until then it is an
  // opaque call, so that no LLVM pass can exploit the UB (fneq rewrites it
  // after the last optimisation).
  //
  // Passes delete the block while nothing branches to it, so it is looked up
  // (or recreated) whenever a fault edge is added.
  // (Everything is looked up by name: globaldce deletes the marker declaration
  // while it is unused.)
  const unsigned fault_idx = static_cast<unsigned>(out_tys.size() - 1);
  auto fault_block = [&]() -> BasicBlock * {
    for (auto &BB : *W)
      if (!BB.empty())
        if (auto *cb = dyn_cast<CallBase>(&BB.front()))
          if (auto *fn = cb->getCalledFunction())
            if (fn->getName() == "fneq.fault") return &BB;
    auto marker = M->getOrInsertFunction("fneq.fault", FunctionType::get(Type::getVoidTy(C), false));
    auto *bb = BasicBlock::Create(C, "fault_exit", W);
    IRBuilder<> fb(bb);
    fb.CreateCall(marker);
    Value *fr = ConstantAggregateZero::get(ret_ty);
    fr = fb.CreateInsertValue(fr, fb.getInt8(1), fault_idx);
    fb.CreateRet(fr);
    return bb;
  };

  // ---------------------------------------------- 5. inline everything
  for (auto &F : *M) {
    if (&F == W || F.isDeclaration()) continue;
    F.setLinkage(GlobalValue::InternalLinkage);
    F.removeFnAttr(Attribute::NoInline);
    F.removeFnAttr(Attribute::OptimizeNone);
    F.removeFnAttr(Attribute::NoDuplicate);
    F.addFnAttr(Attribute::AlwaysInline);
    for (auto &arg : F.args()) arg.removeAttr(Attribute::NoAlias);
  }
  // Static constructors of the semantics module are irrelevant to the lifted code.
  if (auto *ctors = M->getGlobalVariable("llvm.global_ctors")) ctors->eraseFromParent();
  for (auto &G : M->globals())
    if (!G.isDeclaration() && !G.getName().starts_with("llvm."))
      G.setLinkage(GlobalValue::InternalLinkage);
  RunPipeline(*M, "always-inline,globaldce");

  for (auto &I : instructions(*W))
    if (auto *cb = dyn_cast<CallBase>(&I))
      if (auto *fn = cb->getCalledFunction())
        if (!fn->isDeclaration() || fn->getName().starts_with("fneq_trace_"))
          Unsupported("call that could not be inlined: " + fn->getName().str() +
                      " (recursion?)");
  if (g_unsupported.size()) Finish(false, "inlining failed");

  // Lifted locals (BRANCH_TAKEN, RETURN_PC, ...) are zero-initialised so a
  // read-before-write can never observe an uninitialised (poison) value.
  for (auto &I : make_early_inc_range(W->getEntryBlock())) {
    if (auto *a = dyn_cast<AllocaInst>(&I)) {
      if (a == state) continue;
      IRBuilder<> b(a->getNextNode());
      auto sz = a->getAllocationSize(M->getDataLayout());
      if (!sz) { Unsupported("dynamic alloca in lifted code"); continue; }
      b.CreateMemSet(a, b.getInt8(0), sz->getFixedValue(), a->getAlign());
    }
  }
  // Faults raised inside instruction semantics (e.g. DIV by zero) leave
  // through the fault exit.
  // (Look the intrinsic up by name: globaldce may have deleted the declaration
  // that the IntrinsicTable points to.)
  if (auto *error_fn = M->getFunction("__remill_error")) {
    std::vector<CallBase *> errs;
    for (auto *u : error_fn->users())
      if (auto *cb = dyn_cast<CallBase>(u))
        if (cb->getFunction() == W) errs.push_back(cb);
    for (auto *cb : errs) {
      BasicBlock *bb = cb->getParent();
      bb->splitBasicBlock(cb->getIterator());
      bb->getTerminator()->eraseFromParent();
      BranchInst::Create(fault_block(), bb);
      cb->replaceAllUsesWith(Constant::getNullValue(cb->getType()));
      cb->eraseFromParent();
    }
  }
  StripPoison(*W);
  MakeShiftsTotal(*W);
  GuardDivisions(*W, fault_block());
  if (unsigned n = MakeTotal(*W))
    g_notes.push_back(std::to_string(n) + " undef/poison operands in the lifted semantics were "
                      "replaced by zero");
  RunPipeline(*M, kCleanup);
  StripPoison(*W);
  for (int i = 0; i < 4 && CanonicalizeOverflowFlags(*W); ++i) {
    RunPipeline(*M, kCleanup);
    StripPoison(*W);
  }

  // Every remaining __remill_* call is a semantic we do not model.
  for (auto &I : instructions(*W))
    if (auto *cb = dyn_cast<CallBase>(&I))
      if (auto *fn = cb->getCalledFunction())
        if (fn->getName().starts_with("__remill_"))
          Unsupported("unmodelled remill intrinsic " + fn->getName().str());
  for (auto &I : instructions(*W))
    if (auto *a = dyn_cast<AllocaInst>(&I))
      Unsupported("register state was not fully promoted (alloca of " +
                  std::to_string(a->getAllocationSize(M->getDataLayout())->getFixedValue()) +
                  " bytes left)");
  if (g_unsupported.size()) Finish(false, "lowering failed");

  // ----------------------------------- 6. switch to a flat 64-bit layout
  // All State accesses are SSA values now, so the data layout no longer
  // matters for them. The memory model needs pointers wide enough to index
  // the whole address space.
  M->setDataLayout("e-m:e-p:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128");
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  auto &DL = M->getDataLayout();
  (void) DL;

  // ----------------------------------------------- 7. lower memory accesses
  const uint64_t mem_size = abits == 32 ? (1ULL << 32) : (1ULL << 47);
  auto *Mptr = W->getArg(0);
  Mptr->addAttr(Attribute::getWithDereferenceableBytes(C, mem_size));
  auto *i64 = Type::getInt64Ty(C);
  auto *i8 = Type::getInt8Ty(C);

  IRBuilder<> eb(&*W->getEntryBlock().getFirstInsertionPt());
  const uint64_t frame_size = spec.frame_below + spec.frame_above;
  auto *frame = eb.CreateAlloca(ArrayType::get(i8, frame_size), nullptr, "frame");
  // Frame starts with whatever the caller left in memory below its SP.
  auto *esp0_const = ConstantInt::get(addr_ty, spec.esp0);
  // A (possibly new) call to the esp0 marker at the top of the function.
  auto get_esp0 = [&]() -> Value * {
    for (auto &I : W->getEntryBlock())
      if (IsCallTo(&I, esp0_fn)) return &I;
    IRBuilder<> b(&*W->getEntryBlock().getFirstInsertionPt());
    return b.CreateCall(esp0_fn, {}, "esp0");
  };
  // `guarded`: the access already branched to the fault exit unless
  // addr <= mem_size - size, so the index is in bounds as is. Unguarded
  // addresses are masked so that the access can never be out of bounds (UB).
  auto mem_index = [&](IRBuilder<> &b, Value *addr, bool guarded = false) -> Value * {
    Value *idx = b.CreateZExt(addr, i64);
    if (abits == 64 && !guarded) idx = b.CreateAnd(idx, ConstantInt::get(i64, mem_size - 1));
    return b.CreateGEP(i8, Mptr, idx);
  };
  {
    auto *src = mem_index(eb, ConstantInt::get(addr_ty, spec.esp0 - spec.frame_below));
    eb.CreateMemCpy(frame, MaybeAlign(1), src, MaybeAlign(1), frame_size);
  }
  if (spec.frame_visible) {
    // A stack address escapes (an earlier attempt found it). It takes its machine
    // value below: esp0 is the same constant on both sides, so the two functions
    // agree only if the escaped addresses agree (that part of the frame layout is
    // shared). Callees reach the frame through it: around every external call the
    // frame is written to %M at its machine addresses and read back afterwards.
    // At return %M's frame region gets its input contents back, so the frame's
    // final contents stay unobservable.
    g_notes.push_back("a stack address escapes: the frame is visible to callees at its "
                      "machine addresses");
    // Only the window [lo, hi) around the escaped addresses is published: copying
    // the whole frame makes the solver's job much harder.
    int64_t lo = -static_cast<int64_t>(spec.frame_below), hi = static_cast<int64_t>(spec.frame_above);
    if (spec.has_window) {
      lo = std::max(lo, spec.window_lo);
      hi = std::min(hi, spec.window_hi);
    }
    if (hi <= lo) Fail("empty frame window");
    const uint64_t win_size = static_cast<uint64_t>(hi - lo);
    auto *frame_mem = mem_index(eb, ConstantInt::get(addr_ty, spec.esp0 + lo), /*guarded=*/true);
    auto *frame_win = eb.CreateGEP(i8, frame, ConstantInt::get(i64, lo + static_cast<int64_t>(spec.frame_below)));
    auto *frame_in = eb.CreateAlloca(ArrayType::get(i8, win_size), nullptr, "frame_in");
    eb.CreateMemCpy(frame_in, MaybeAlign(1), frame_mem, MaybeAlign(1), win_size);
    std::vector<CallInst *> outs, backs;
    std::vector<ReturnInst *> rets;
    for (auto &I : instructions(*W)) {
      if (auto *ci = dyn_cast<CallInst>(&I))
        if (auto *fn = ci->getCalledFunction()) {
          if (fn->getName() == "fneq.frame_out") outs.push_back(ci);
          if (fn->getName() == "fneq.frame_back") backs.push_back(ci);
        }
      if (auto *ri = dyn_cast<ReturnInst>(&I)) rets.push_back(ri);
    }
    for (auto *ci : outs) {
      // The callee sees the frame at its machine addresses; its own return-address
      // slot (a code address, different in every build) reads as zero.
      IRBuilder<> b(ci);
      b.CreateMemCpy(frame_mem, MaybeAlign(1), frame_win, MaybeAlign(1), win_size);
      // If the window covers the callee's return-address slot, that slot reads as
      // zero (it holds a code address, different in every build).
      Value *ra_rel = b.CreateSub(ci->getArgOperand(0), ConstantInt::get(addr_ty, spec.esp0 + lo));
      Value *ra_in = b.CreateICmpULE(ra_rel, ConstantInt::get(addr_ty, win_size - abytes));
      Value *ra_ptr = b.CreateGEP(i8, frame_mem, b.CreateZExt(ra_rel, i64));
      Value *old = b.CreateLoad(addr_ty, b.CreateSelect(ra_in, ra_ptr, frame_mem));
      cast<LoadInst>(old)->setAlignment(Align(1));
      b.CreateStore(b.CreateSelect(ra_in, ConstantInt::get(addr_ty, 0), old),
                    b.CreateSelect(ra_in, ra_ptr, frame_mem))->setAlignment(Align(1));
      ci->eraseFromParent();
    }
    for (auto *ci : backs) {
      IRBuilder<> b(ci);
      b.CreateMemCpy(frame_win, MaybeAlign(1), frame_mem, MaybeAlign(1), win_size);
      ci->eraseFromParent();
    }
    for (auto *ri : rets) {
      IRBuilder<> b(ri);
      b.CreateMemCpy(frame_mem, MaybeAlign(1), frame_in, MaybeAlign(1), win_size);
    }
  }

  std::vector<GlobalVariable *> ro_globals;
  for (auto &r : spec.rodata) {
    auto *init = ConstantDataArray::get(C, r.bytes);
    ro_globals.push_back(new GlobalVariable(*M, init->getType(), true,
                                            GlobalValue::PrivateLinkage, init,
                                            "ro_" + utohexstr(r.addr)));
  }

  // Build the pointer for an access of `bytes` at machine address `addr`.
  // Translate global addresses of this side into the shared (side-A) space.
  auto remap_addr = [&](IRBuilder<> &b, Value *addr) -> Value * {
    if (spec.remap.empty()) return addr;
    if (auto *c = dyn_cast<ConstantInt>(addr)) {
      uint64_t a = c->getZExtValue();
      for (auto &r : spec.remap)
        if (a >= r.from && a < r.from + r.size)
          return ConstantInt::get(addr_ty, r.to + (a - r.from));
      return addr;
    }
    Value *out = addr;
    for (auto &r : spec.remap) {
      Value *rel = b.CreateSub(addr, ConstantInt::get(addr_ty, r.from));
      Value *in = b.CreateICmpULT(rel, ConstantInt::get(addr_ty, r.size));
      out = b.CreateSelect(in, b.CreateAdd(rel, ConstantInt::get(addr_ty, r.to)), out);
    }
    return out;
  };

  auto lower_pointer = [&](IRBuilder<> &b, Value *addr, unsigned bytes,
                           bool is_read) -> Value * {
    if (!DependsOn(addr, esp0_fn)) addr = remap_addr(b, addr);
    // Constant addresses.
    if (auto *c = dyn_cast<ConstantInt>(addr)) {
      uint64_t a = c->getZExtValue();
      if (is_read)
        for (size_t k = 0; k < spec.rodata.size(); ++k) {
          auto &r = spec.rodata[k];
          if (a >= r.addr && a + bytes <= r.addr + r.bytes.size())
            return b.CreateGEP(i8, ro_globals[k], ConstantInt::get(i64, a - r.addr));
        }
      return mem_index(b, addr, /*guarded=*/true);
    }
    if (DependsOn(addr, esp0_fn)) {
      // Stack access: offset relative to esp0, normalised by instcombine.
      Value *off = b.CreateSub(addr, get_esp0());
      Value *rel = b.CreateAdd(off, ConstantInt::get(addr_ty, spec.frame_below));
      Value *in_frame = b.CreateICmpULE(
          rel, ConstantInt::get(addr_ty, frame_size - bytes));
      Value *fptr = b.CreateGEP(i8, frame, b.CreateZExt(rel, i64));
      Value *mptr = mem_index(b, b.CreateAdd(esp0_const, off));
      return b.CreateSelect(in_frame, fptr, mptr);
    }
    // Symbolic non-stack address (range-checked by the caller): may land in a
    // constant table.
    Value *p = mem_index(b, addr, /*guarded=*/true);
    if (is_read)
      for (size_t k = 0; k < spec.rodata.size(); ++k) {
        auto &r = spec.rodata[k];
        if (r.bytes.size() < bytes) continue;
        Value *rel = b.CreateSub(addr, ConstantInt::get(addr_ty, r.addr));
        Value *in = b.CreateICmpULE(rel, ConstantInt::get(addr_ty, r.bytes.size() - bytes));
        Value *rp = b.CreateGEP(i8, ro_globals[k], b.CreateZExt(rel, i64));
        p = b.CreateSelect(in, rp, p);
      }
    return p;
  };

  for (int round = 0;; ++round) {
    if (round > 32) Fail("memory lowering did not converge");
    std::vector<CallInst *> pending;
    for (auto &I : instructions(*W))
      if (auto *ci = dyn_cast<CallInst>(&I))
        if (auto *fn = ci->getCalledFunction())
          if (fn->getName().starts_with("fneq.read.") ||
              fn->getName().starts_with("fneq.write."))
            pending.push_back(ci);
    if (pending.empty()) break;
    bool progress = false;
    bool final_round = false;
    // An access whose address depends on a pending read could be a stack
    // pointer that went through memory; lower it only after that read has
    // been resolved (and SROA had a chance to turn it into a value).
    std::vector<CallInst *> ready;
    for (auto *ci : pending) {
      std::set<Value *> seen;
      if (!DependsOnPendingRead(ci->getArgOperand(0), seen)) ready.push_back(ci);
    }
    if (ready.empty()) { ready = pending; final_round = true; }
    for (auto *ci : ready) {
      auto name = ci->getCalledFunction()->getName();
      bool is_read = name.starts_with("fneq.read.");
      Value *addr = ci->getArgOperand(0);
      {
        unsigned bytes = is_read ? ci->getType()->getIntegerBitWidth() / 8
                                 : ci->getArgOperand(1)->getType()->getIntegerBitWidth() / 8;
        // An access outside the user address range (x86-64: canonical lower
        // half; x86: wrapping past 4 GiB) faults.
        if (!DependsOn(addr, esp0_fn)) {
          IRBuilder<> g(ci);
          Value *bad = g.CreateICmpUGT(addr, ConstantInt::get(addr_ty, mem_size - bytes));
          auto *cbad = dyn_cast<ConstantInt>(bad);
          if (!cbad || !cbad->isZero()) BranchToFault(bad, ci, fault_block());
        }
      }
      IRBuilder<> b(ci);
      if (is_read) {
        auto *ty = ci->getType();
        unsigned bytes = ty->getIntegerBitWidth() / 8;
        auto *ld = b.CreateLoad(ty, lower_pointer(b, addr, bytes, true));
        ld->setAlignment(Align(1));
        ci->replaceAllUsesWith(ld);
      } else {
        auto *val = ci->getArgOperand(1);
        unsigned bytes = val->getType()->getIntegerBitWidth() / 8;
        auto *st = b.CreateStore(val, lower_pointer(b, addr, bytes, false));
        st->setAlignment(Align(1));
      }
      ci->eraseFromParent();
      progress = true;
    }
    if (final_round)
      g_notes.push_back("memory accesses through values loaded from non-promotable memory "
                        "were classified as non-stack");
    if (!progress) break;
    RunPipeline(*M, kCleanup);
  }

  // Calls through an import slot: the slot holds the import's stub address (a
  // constant table), so the indirect call's target folded to that constant. Such a
  // call is a call of the import: same arguments without the target.
  RunPipeline(*M, kCleanup);
  if (auto *ind = M->getFunction("ext." + spec.indirect.name); ind && spec.has_indirect) {
    std::vector<CallInst *> calls;
    for (auto *u : ind->users())
      if (auto *ci = dyn_cast<CallInst>(u)) calls.push_back(ci);
    for (auto *ci : calls) {
      auto *k = dyn_cast<ConstantInt>(ci->getArgOperand(0));
      if (!k) continue;
      auto it = spec.callees.find(k->getZExtValue());
      if (it == spec.callees.end()) continue;
      const Callee &imp = it->second;
      std::vector<Type *> tys;
      std::vector<Value *> args;
      for (unsigned i = 1; i < ci->arg_size(); ++i) {
        tys.push_back(ci->getArgOperand(i)->getType());
        args.push_back(ci->getArgOperand(i));
      }
      if (imp.args.size() != args.size() || imp.results != spec.indirect.results) {
        Unsupported("call through the import slot of " + imp.name + " with a different model");
        continue;
      }
      auto callee = M->getOrInsertFunction("ext." + imp.name, FunctionType::get(ci->getType(), tys, false));
      if (auto *fn = dyn_cast<Function>(callee.getCallee())) {
        fn->setDoesNotThrow();
        fn->setWillReturn();
      }
      IRBuilder<> b(ci);
      auto *nc = b.CreateCall(callee, args);
      ci->replaceAllUsesWith(nc);
      ci->eraseFromParent();
    }
    if (g_unsupported.size()) Finish(false, "unsupported import call");
  }

  // Remaining uses of esp0 mean a stack address escaped (passed to a callee,
  // stored to non-stack memory, returned, compared, ...). Without frame_visible
  // this attempt stops and the orchestrator lifts again with it set.
  RunPipeline(*M, kCleanup);
  for (auto *u : make_early_inc_range(esp0_fn->users())) {
    auto *call = cast<CallBase>(u);
    if (!call->use_empty() && !spec.frame_visible) {
      std::string s;
      raw_string_ostream os(s);
      call->user_back()->print(os);
      for (auto *use : call->users()) {
        auto *bo = dyn_cast<BinaryOperator>(use);
        auto *c = bo && bo->getOpcode() == Instruction::Add ? dyn_cast<ConstantInt>(bo->getOperand(1)) : nullptr;
        bool plain = c && bo->getOperand(0) == call;
        if (plain)
          for (auto *u2 : bo->users())
            if (!isa<CallBase>(u2) && !isa<StoreInst>(u2) && !isa<InsertValueInst>(u2))
              plain = false;  // further arithmetic: the escaped range is not known
        if (plain)
          g_escapes.push_back(c->getSExtValue());
        else if (isa<CallBase>(use) || isa<StoreInst>(use) || isa<InsertValueInst>(use))
          g_escapes.push_back(0);  // esp0 itself
        else
          g_escape_unknown = true;
      }
      Unsupported("address of a stack slot escapes: " + s);
      Finish(false, "stack address escapes");
    }
    call->replaceAllUsesWith(esp0_const);
    call->eraseFromParent();
  }
  if (spec.frame_visible) RunPipeline(*M, kCleanup);

  if (LowerFPExtToFP80(*W)) RunPipeline(*M, kCleanup);
  if (NarrowUnsignedDivisions(*W)) RunPipeline(*M, kCleanup);
  RunPipeline(*M, "globaldce");
  StripPoison(*W);
  Audit(*W);
  W->setName(spec.name);
  if (verifyModule(*M, &errs())) Fail("lowered module does not verify");

  std::error_code ec;
  raw_fd_ostream os(FLAGS_out, ec);
  if (ec) Fail("cannot write " + FLAGS_out);
  M->print(os, nullptr);
  os.close();
  Finish(true);
}
