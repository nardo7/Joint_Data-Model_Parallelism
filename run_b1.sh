#!/bin/bash
cd "$(dirname "$0")" || return
START_TIME=$SECONDS
# activate current .venv environment
source ../../.venv/bin/activate

for ((i=0; i<3; i=i+1))
do
    touch "b1_out$i.txt"
    (sleep 1; python3 -u "microbatches.py" $i "B1">"b1_out$i.txt") &
done

wait
echo "Elapsed time (s): $((SECONDS - START_TIME))"
