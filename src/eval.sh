#!/bin/bash

if [ "$1" == "0" ]; then
    # Stage 1 Resize ratio: 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
    for r in $(seq 0.20 0.05 0.50); do
        python run_gui_actor.py $1 --r $r --th 0.03 --p 0
        python run_gui_actor.py $1 --r $r --th 0.03 --p 10
        python run_gui_actor.py $1 --r $r --th 0.03 --p 20
        python run_gui_actor.py $1 --r $r --th 0.03 --p 30
        python run_gui_actor.py $1 --r $r --th 0.03 --p 40
        python run_gui_actor.py $1 --r $r --th 0.03 --p 50
    done

elif [ "$1" == "1" ]; then
    for r in $(seq 0.20 0.05 0.50); do
        python run_gui_actor.py $1 --r $r --th 0.05 --p 0
        python run_gui_actor.py $1 --r $r --th 0.05 --p 10
        python run_gui_actor.py $1 --r $r --th 0.05 --p 20
        python run_gui_actor.py $1 --r $r --th 0.05 --p 30
        python run_gui_actor.py $1 --r $r --th 0.05 --p 40
        python run_gui_actor.py $1 --r $r --th 0.05 --p 50
    done

elif [ "$1" == "2" ]; then
    for r in $(seq 0.20 0.05 0.50); do
        python run_gui_actor.py $1 --r $r --th 0.07 --p 0
        python run_gui_actor.py $1 --r $r --th 0.07 --p 10
        python run_gui_actor.py $1 --r $r --th 0.07 --p 20
        python run_gui_actor.py $1 --r $r --th 0.07 --p 30
        python run_gui_actor.py $1 --r $r --th 0.07 --p 40
        python run_gui_actor.py $1 --r $r --th 0.07 --p 50
    done

elif [ "$1" == "3" ]; then
    for r in $(seq 0.20 0.05 0.50); do
        python run_gui_actor.py $1 --r $r --th 0.10 --p 0
        python run_gui_actor.py $1 --r $r --th 0.10 --p 10
        python run_gui_actor.py $1 --r $r --th 0.10 --p 20
        python run_gui_actor.py $1 --r $r --th 0.10 --p 30
        python run_gui_actor.py $1 --r $r --th 0.10 --p 40
        python run_gui_actor.py $1 --r $r --th 0.10 --p 50
    done

else
    echo "Usage: ./eval.sh [0|1|2|3]"
fi