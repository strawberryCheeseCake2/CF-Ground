import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 전체 폰트 사이즈 전역 설정
plt.rc('font', size=24)
plt.rc('axes', titlesize=24, labelsize=24)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('legend', fontsize=22)

# 데이터 정리
resize_ratios = ["s:0.3","s:0.4","s:0.5","s:0.6","s:0.7","s:0.8","s:0.9","s:1.0"]

acc_stage1 = [72.25, 80.58, 85.3, 88.05, 89.78, 90.09, 89.86, 90.8]
tflops_stage1 = [4.8, 8.42, 13.58, 20.74, 30.48, 43.39, 60.26, 71.53]

acc_stage2 = [88.36, 88.29, 89.62, 90.64, 89.39, 90.41, 90.25, 90.09]
tflops_stage2 = [8.79, 11.2, 15.9, 22.92, 32.5, 45.42, 62.24, 73.53]

acc_stage3 = [87.97, 88.92, 91.43, 91.51, 91.27, 91.35, 91.98, 91.75]
tflops_stage3 = [8.79, 11.2, 15.9, 22.92, 32.5, 45.42, 62.24, 73.53]

# 원래 세 개 Stage 라인
plt.figure(figsize=(10,6))
plt.plot(tflops_stage1, acc_stage1, '^-', color='blue', label="Global Stage")
plt.plot(tflops_stage2, acc_stage2, 's-', color='green', label="Global + Local Stage")
plt.plot(tflops_stage3, acc_stage3, 'o-', color='red', label="Global + Local + Fusion Stage")

# 같은 resize끼리 연결선
for i in range(len(resize_ratios)):
    xs = [tflops_stage1[i], tflops_stage2[i], tflops_stage3[i]]
    ys = [acc_stage1[i], acc_stage2[i], acc_stage3[i]]
    plt.plot(xs, ys, color='black', linestyle='--', alpha=0.9, zorder=0)

# 텍스트 라벨 (Stage3 기준)
for x, y, r in zip(tflops_stage3, acc_stage3, resize_ratios):
    plt.text(x-0.5, y+1, f"{r}", color='black', ha='center', fontsize=12)

plt.xlabel("TFLOPs")
plt.ylabel("Accuracy (%)")
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.8)
plt.ylim(65, 95)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(20))

plt.savefig("pdf/ablation_stage.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("png/ablation_stage.png", dpi=300, bbox_inches='tight', pad_inches=0)

print(f"📈 Saved at ./pdf/ablation_stage.pdf")
print(f"📈 Saved at ./png/ablation_stage.png")
