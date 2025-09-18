#!/bin/bash

if [ "$1" == "0" ]; then
    for r in $(seq 0.30 0.10 1.00); do
        python run_gui_actor.py $1 --r $r --th 0.11 --p 0
    done

elif [ "$1" == "1" ]; then
    for r in $(seq 0.30 0.10 1.00); do
        python run_gui_actor.py $1 --r $r --th 0.12 --p 0
    done

elif [ "$1" == "2" ]; then
    for r in $(seq 0.30 0.10 1.00); do
        python run_gui_actor.py $1 --r $r --th 0.20 --p 0
    done

elif [ "$1" == "3" ]; then
    for r in $(seq 0.30 0.10 1.00); do
        python run_gui_actor.py $1 --r $r --th 0.20 --p 0
    done

else
    echo "Usage: ./eval.sh [0|1|2|3]"
fi