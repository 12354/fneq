"""Each external edge of a conditional jump has its own tail model."""
import json
import pytest
pytest.importorskip('angr')
from fneq import binary as binmod,bounding,pipeline


@pytest.mark.parametrize('shuffle',[False,True])
@pytest.mark.parametrize('swap',[False,True])
def test_conditional_tail_edges_are_not_order_dependent(tmp_path,monkeypatch,shuffle,swap):
    binaries={}
    for tag in ('a','b'):
        flipped=tag=='b' and swap
        taken,fall=(0x1040,0x1060) if flipped else (0x1060,0x1040)
        # Preserve an observable int argument, then branch to either tail target.
        root=bytes.fromhex('89d2 4885c9 74')+bytes([taken-0x1007])
        thunk=b'\xe9'+(fall-0x100C).to_bytes(4,'little',signed=True)
        data=root+thunk+b'\xcc'*(0x40-len(root)-len(thunk))+bytes.fromhex('31c0 c3')
        data+=b'\xcc'*(0x60-len(data))+bytes.fromhex('89d0 c3')
        path=tmp_path/(tag+'.bin');path.write_bytes(data)
        b=binmod.load(str(path),0x1000,'x64');bounding.angr_project(b);b.fmt='pe'
        b.pdata_ranges=[(0x1000,0x1007,0),(0x1007,0x100C,0),(0x1040,0x1043,0),(0x1060,0x1063,0)]
        binaries[tag]=b
    original=bounding._bound
    def reordered(binary,entry):
        info=original(binary,entry)
        if shuffle and binary is binaries['b'] and entry==0x1000:
            info.calls.reverse()
        return info
    monkeypatch.setattr(bounding,'_bound',reordered)
    monkeypatch.setattr(pipeline,'_load_binary',lambda side:binaries[side.path])
    corr=tmp_path/'corr.json'
    corr.write_text(json.dumps({'callees':[{'a':'0x1040','b':'0x1040','name':'zero'},
                                          {'a':'0x1060','b':'0x1060','name':'identity'}]}))
    opts=pipeline.Options(pipeline.SideOpts('a','0x1000',cc='ms'),pipeline.SideOpts('b','0x1000',cc='ms'),
                          backend='lockstep',no_alias=True,ret='i32',corr=str(corr))
    rep=pipeline.analyse(opts)
    assert rep.verdict==('UNKNOWN' if swap else 'EQUIVALENT'),rep.reason
