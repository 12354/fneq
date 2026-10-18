"""Explicit complete signatures work without guessing an unresolved body's arity."""
import json
import pytest
pytest.importorskip('angr')
from fneq import binary as binmod,bounding,pipeline


@pytest.mark.parametrize('same',[False,True])
def test_complete_signature_checks_changed_argument(tmp_path,monkeypatch,same):
    binaries={}
    for tag,value in [('a',7),('b',7 if same else 8)]:
        root=bytes.fromhex('4883ec28 ba')+value.to_bytes(4,'little')
        root+=b'\xe8'+(0x40-len(root)-5).to_bytes(4,'little')+bytes.fromhex('4883c428 c3')
        path=tmp_path/(tag+'.bin');path.write_bytes(root+b'\xcc'*(0x40-len(root))+bytes.fromhex('ffe0'))
        b=binmod.load(str(path),0x1000,'x64');bounding.angr_project(b);b.fmt='pe'
        b.pdata_ranges=[(0x1000,0x1000+len(root),0),(0x1040,0x1042,0)]
        binaries[tag]=b
    corr=tmp_path/'corr.json'
    corr.write_text(json.dumps({'callees':[{'a':'0x1040','b':'0x1040','name':'dispatch'}],
        'callee_contracts':{'dispatch':{'reg_args':['RCX','RDX:32'],'stack_args':0,'pop':0}}}))
    monkeypatch.setattr(pipeline,'_load_binary',lambda side:binaries[side.path])
    opts=pipeline.Options(pipeline.SideOpts('a','0x1000',cc='ms'),pipeline.SideOpts('b','0x1000',cc='ms'),
                          backend='lockstep',no_alias=True,ret='i32',corr=str(corr))
    rep=pipeline.analyse(opts)
    assert rep.verdict==('EQUIVALENT' if same else 'UNKNOWN'),rep.reason
    assert any('declared complete' in n for n in rep.data['notes'])
