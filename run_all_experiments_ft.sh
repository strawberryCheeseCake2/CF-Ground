#!/bin/bash

# GUI Actor 4가지 실험 자동화 스크립트
echo "=== GUI Actor 4가지 실험 시작 ==="
echo "실험 시작 시간: $(date)"
echo ""

# 실험 1: EARLY_EXIT=True, MAX_PIXELS=1280*28*28
echo "🚀 [1/4] Early Exit ON, MAX_PIXELS=1003520"
echo "시작: $(date)"
python run_gui_actor_ft.py --max_pixels 1003520
echo "완료: $(date)"
echo ""

# 실험 2: EARLY_EXIT=False, MAX_PIXELS=1280*28*28  
echo "❇️ [2/4] Early Exit OFF, MAX_PIXELS=1003520"
echo "시작: $(date)"
python run_gui_actor_ft.py --no_early_exit --max_pixels 1003520
echo "완료: $(date)"
echo ""

# 실험 3: EARLY_EXIT=True, MAX_PIXELS=3211264
echo "🚀 [3/4] Early Exit ON, MAX_PIXELS=3211264"
echo "시작: $(date)"
python run_gui_actor_ft.py --max_pixels 3211264
echo "완료: $(date)"
echo ""

# 실험 4: EARLY_EXIT=False, MAX_PIXELS=3211264
echo "❇️ [4/4] Early Exit OFF, MAX_PIXELS=3211264"
echo "시작: $(date)"
python run_gui_actor_ft.py --no_early_exit --max_pixels 3211264
echo "완료: $(date)"
echo ""

echo "=== 모든 실험 완료 ==="
echo "종료 시간: $(date)"
