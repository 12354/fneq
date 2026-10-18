"""Different compiler joins must not truncate one side before a shared call."""
import pytest
pytest.importorskip('angr')
from fneq import binary as binmod,bounding,lockstep,pipeline


def root(phi,value,extra=False,memory=None):
    code=bytearray(bytes.fromhex('4883ec28 b901000000'))
    labels,fixups={},[]
    def branch(op,target):
        code.extend(bytes.fromhex(op));fixups.append((len(code),target));code.extend(bytes(4))
    def label(name):labels[name]=0x1000+len(code)
    branch('e8',0x1200)
    code.extend(bytes.fromhex('85c0'))
    if phi:
        branch('0f84','zero');code.extend(bytes.fromhex('41ba01000000'));branch('e9','phi')
        label('zero');code.extend(bytes.fromhex('4531d2'))
        label('phi');code.extend(bytes.fromhex('4585d2'))
    branch('0f84','fail')
    code.extend(b'\xba'+value.to_bytes(4,'little'));branch('e8',0x1210)
    if extra:
        code.extend(b'\xba'+value.to_bytes(4,'little'));branch('e8',0x1210)
    if memory is not None:
        code.extend(bytes.fromhex('49bb0000003000000000 41c703')+memory.to_bytes(4,'little'))
    branch('e9','end')
    label('fail');code.extend(bytes.fromhex('31c0'))
    label('end');code.extend(bytes.fromhex('4883c428 c3'))
    for pos,target in fixups:
        dest=labels[target] if isinstance(target,str) else target
        code[pos:pos+4]=(dest-0x1000-pos-4).to_bytes(4,'little',signed=True)
    return bytes(code)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('changed',[False,True])
def test_asymmetric_join_cuts(tmp_path,monkeypatch,reverse,changed):
    run_case(tmp_path,monkeypatch,reverse,changed,False)


def test_extra_call_is_not_hidden_by_join_extension(tmp_path,monkeypatch):
    run_case(tmp_path,monkeypatch,False,False,True)


@pytest.mark.parametrize('different',[False,True])
def test_memory_checks_survive_join_extension(tmp_path,monkeypatch,different):
    run_case(tmp_path,monkeypatch,False,False,False,(1,2 if different else 1))


def run_case(tmp_path,monkeypatch,reverse,changed,extra,memory=None):
    originals=[root(False,7,memory=memory[0] if memory else None),
               root(True,8 if changed else 7,extra,memory=memory[1] if memory else None)]
    if reverse:originals.reverse()
    binaries={}
    for tag,code in zip(('a','b'),originals):
        data=code+b'\xcc'*(0x200-len(code))+bytes.fromhex('89c8 c3')
        data+=b'\xcc'*(0x210-len(data))+bytes.fromhex('89d0 c3')
        path=tmp_path/(tag+'.bin');path.write_bytes(data)
        b=binmod.load(str(path),0x1000,'x64');bounding.angr_project(b);b.fmt='pe'
        b.pdata_ranges=[(0x1000,0x1000+len(code),0),(0x1200,0x1203,0),(0x1210,0x1213,0)]
        binaries[tag]=b
    monkeypatch.setattr(pipeline,'_load_binary',lambda side:binaries[side.path])
    # Exercise the same join cuts used after the path-growth limit is reached.
    original=lockstep.Checker._run_segment
    def segmented(self,s,start,regs,stack,words,escaped,visit,cuts=None):
        return original(self,s,start,regs,stack,words,escaped,visit,
                        s.heads|s.joins if cuts is None else cuts)
    monkeypatch.setattr(lockstep.Checker,'_run_segment',segmented)
    opts=pipeline.Options(pipeline.SideOpts('a','0x1000',cc='ms'),pipeline.SideOpts('b','0x1000',cc='ms'),
                          backend='lockstep',no_alias=True,ret='i32')
    rep=pipeline.analyse(opts)
    differs=changed or extra or (memory and memory[0]!=memory[1])
    assert rep.verdict==('UNKNOWN' if differs else 'EQUIVALENT'),rep.reason
    assert rep.data['lockstep'].get('extended_join_cuts'),rep.data['lockstep']
