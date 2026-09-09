#!/usr/bin/env bash
# Run / resume the case in bounded chunks (for environments with a wall-clock
# cap per command). Each call restarts from the newest checkpoint and appends
# to run.log; call repeatedly until the log reports "finalized".
set -u
EXE=${EXE:-../../Build_Gnumake/ExaGOOP2d.gnu.ex}
INP=Inputs_2D_Plate_Under_Gravity.inp
TAG=2D_Plate_Under_Gravity
LIMIT=${LIMIT:-38}
CHK=$(ls -d Solution/checkpoint_files/$TAG/chk?????? 2>/dev/null | sort | tail -1)
if [ -n "$CHK" ]; then
  timeout $LIMIT $EXE $INP amr.restart_checkfile="\"$CHK\"" mpm.write_output_time=${WOT:-0.005} >> run.log 2>&1
else
  timeout $LIMIT $EXE $INP mpm.write_output_time=${WOT:-0.005} >> run.log 2>&1
fi
echo "exit=$? last_chk=$(basename "${CHK:-none}") $(grep -E 'Writing outputs' run.log | tail -1)"
