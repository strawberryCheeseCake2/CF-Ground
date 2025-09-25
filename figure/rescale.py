import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 전체 폰트 사이즈 조정
plt.rc('font', size=24)
plt.rc('axes', titlesize=24, labelsize=24)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('legend', fontsize=23)

resize_ratios = ["s:0.3","s:0.4","s:0.5","s:0.6","s:0.7","s:0.8","s:0.9","s:1.0"]

acc_2b = [70.68,80.42,84.51,86.79,86.95,88.05,88.76,89.31]
tflops_2b = [3.13,5.92,10.01,15.83,23.94,34.9,49.44,59.21]

acc_2b_gold = [86.4,87.89,87.97,88.68,88.99,89.39,88.36,89.07]
tflops_2b_gold = [6.09,7.87,11.62,17.24,25.26,36.16,50.66,60.41]

acc_3b = [72.09,80.58,85.46,87.89,89.62,90.02,89.47,90.72]
tflops_3b = [4.69,8.31,13.46,20.61,30.35,43.25,60.11,71.38]

tflops_3b_gold = [8.79,11.2,15.9,22.92,32.5,45.42,62.24,73.53]
acc_3b_gold = [87.97,88.92,91.43,91.51,91.27,91.35,91.98,91.75]

plt.figure(figsize=(10,6))
plt.plot(tflops_3b, acc_3b, '^--', color='blue', label="GUI-Actor-3B")
plt.plot(tflops_2b, acc_2b, 's--', color='green', label="GUI-Actor-2B")
plt.plot(tflops_3b_gold, acc_3b_gold, 'o-', color='red', label="GUI-Actor-3B + GOLD")
plt.plot(tflops_2b_gold, acc_2b_gold, 'D-', color='orange', label="GUI-Actor-2B + GOLD")

for x, y, r in zip(tflops_3b, acc_3b, resize_ratios):
    plt.text(x, y+0.5, f"{r}", color='blue', ha='center', fontsize=12)

for x, y, r in zip(tflops_2b, acc_2b, resize_ratios):
    plt.text(x, y-1.5, f"{r}", color='green', ha='center', fontsize=12)

for x, y, r in zip(tflops_3b_gold, acc_3b_gold, resize_ratios):
    plt.text(x, y+1, f"{r}", color='red', ha='center', fontsize=12)

for x, y, r in zip(tflops_2b_gold, acc_2b_gold, resize_ratios):
    plt.text(x, y-1.5, f"{r}", color='orange', ha='center', fontsize=12)

plt.xlabel("TFLOPs")
plt.ylabel("Accuracy (%)")
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(65, 95)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(20))

plt.savefig("pdf/ablation_resize_ratio.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("png/ablation_resize_ratio.png", dpi=300, bbox_inches='tight', pad_inches=0)

print(f"📈 Saved at ./pdf/ablation_resize_ratio.pdf")
print(f"📈 Saved at ./png/ablation_resize_ratio.png")