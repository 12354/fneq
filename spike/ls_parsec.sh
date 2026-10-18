#!/bin/bash
# usage: ls_parsec.sh ORIG_ADDR RE_ADDR RET [extra fneq args]  (lockstep stage only)
PQ="C:/Users/Marcel/AppData/Local/Temp/claude/E--repos-Reverse-Engineering/1bba245d-208a-4814-997e-9d8d420ccc08/scratchpad/pq"
a=$1; b=$2; r=$3; shift 3
MSYS_NO_PATHCONV=1 timeout 400 docker run --rm \
  -v "C:/Program Files/Parsec/verified/parsecd.dll:/in/orig.dll:ro" \
  -v "E:/repos/Reverse Engineering/parsec_re/build/src/parsecd.dll:/in/re.dll:ro" \
  -v "$PQ:/pq" -v "E:/repos/Reverse Engineering/fneq:/work:ro" -e PYTHONPATH=/work fneq \
  bash -c "timeout 300 python3 -m fneq /in/orig.dll:$a /in/re.dll:$b --ret $r --corr /pq/corr_min.json --backend lockstep --json $* > /pq/last.json; python3 -c \"
import json; d=json.load(open('/pq/last.json')); l=d.get('lockstep',{})
print(d['verdict'],'|',d['reason'][:300]); print({k:l.get(k) for k in ('segment_pairs','visits','queries','obligations_proved','seconds')})
print('diff:',l.get('first_difference')); print('stopped:',l.get('stopped_at')); print(d.get('traceback','')[-2000:])\""
