#!/bin/bash

if [ "$1" == "0" ]; then
    python run_gui_actor.py $1 --r 0.3 --e 0.7
    python run_gui_actor.py $1 --r 0.4 --e 0.7
    python run_gui_actor.py $1 --r 0.5 --e 0.7
    python run_gui_actor.py $1 --r 0.6 --e 0.7

elif [ "$1" == "1" ]; then
    python run_gui_actor.py $1 --r 0.7 --e 0.7
    python run_gui_actor.py $1 --r 0.8 --e 0.7

elif [ "$1" == "2" ]; then
    python run_gui_actor.py $1 --r 0.9 --e 0.7


elif [ "$1" == "3" ]; then
    python run_gui_actor.py $1 --r 1.0 --e 0.7
    # for r in $(seq 0.30 0.10 1.00); do
    #     python run_gui_actor.py $1 --r $r
    # done

else
    echo "Usage: ./eval.sh [0|1|2|3]"
fi