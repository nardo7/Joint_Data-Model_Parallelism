#!/bin/bash
cd "$(dirname "$0")" || return
START_TIME=$SECONDS
# activate current .venv environment
conda init zsh
source ~/.zshrc
conda activate JDMP

for ((i=0; i<6; i=i+1))
do
    touch "b2_out$i.txt"
    (sleep 1; python3 -u "microbatches.py" $i "B2">"b2_out$i.txt") &
done

wait
echo "Elapsed time (s): $((SECONDS - START_TIME))"
