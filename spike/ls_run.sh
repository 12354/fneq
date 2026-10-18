#!/bin/bash
# usage: ls_run.sh BIN FUNC_A FUNC_B RET [BIN_B] [extra args]
f=$1; a=$2; b=$3; r=$4; fb=${5:-$1}; shift $(( $# < 5 ? $# : 5 ))
timeout 300 python -m fneq $f:$a $fb:$b --ret $r --backend lockstep --json "$@" 2>&1 | python -c "
import json,sys; t=sys.stdin.read()
try: d=json.loads(t)
except Exception: print(t[-3000:]); sys.exit()
l=d.get('lockstep',{})
print(d['verdict'],'|',l.get('reason'),'| pairs',l.get('segment_pairs'),'visits',l.get('visits'),'q',l.get('queries'),'t',l.get('seconds'))
if l.get('first_difference'): print('  diff:',l['first_difference'])
if l.get('stopped_at'): print('  stopped at',l['stopped_at'])
if d.get('traceback'): print(d['traceback'][-2500:])
"
