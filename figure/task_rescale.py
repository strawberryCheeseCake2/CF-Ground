import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 전역 폰트 설정
plt.rc('font', size=28)
plt.rc('axes', titlesize=28, labelsize=28)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)
plt.rc('legend', fontsize=22)


# 데이터 정리
# resize_ratios = ["s:0.3", "s:0.4", "s:0.5", "s:0.6", "s:0.7", "s:0.8", "s:0.9", "s:1.0"]
resize_ratios = ["30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]

acc_mobile_gold    = [91.82,91.82,93.41,92.42,93.01,92.61,92.42,93.61]
tflops_mobile_gold = [9.15,11.97,17.13,24.69,34.87,48.82,67.61,84.53]
acc_mobile_vanilla    = [89.0, 91.4, 92.61, 91.02, 93.01, 92.01, 92.01,92.41]
tflops_mobile_vanilla = [5.16, 8.70, 14.21, 21.77, 32.21, 46.12, 64.86,81.78]

acc_web_gold    = [85.35,88.33,90.39,90.16,89.24,89.7,91.3,91.3]
tflops_web_gold = [9.47,13.41,19.58,28.73,41.46,58.53,80.75,89.26]
acc_web_vanilla    = [74.14,81.23,83.75,88.10,87.64,88.78,88.33,89.01]
tflops_web_vanilla = [5.56,10.58,17.24,26.59,39.38,56.44,78.77,87.24]

acc_desktop_gold    = [85.63,85.33,89.82,91.92,91.32,91.62,92.22,89.52]
tflops_desktop_gold = [7.37,7.14,9.23,12.64,17.23,23.16,29.96,36.44]
acc_desktop_vanilla = [44.01,63.47,76.94,82.93,87.12,88.62,87.12,90.42]
tflops_desktop_vanilla = [2.84,4.73,7.36,11.03,15.73,21.70,28.55,35.03]

# ------------------ Mobile ------------------
plt.figure(figsize=(10,6))
plt.plot(tflops_mobile_vanilla, acc_mobile_vanilla, '^--', linestyle=(0, (3, 3)), color='blue', label="Mobile (GUI-Actor-3B)")
plt.plot(tflops_mobile_gold, acc_mobile_gold, 'o-', color='blue', label="Mobile (GUI-Actor-3B + GOLD)")

for x, y, r in zip(tflops_mobile_gold, acc_mobile_gold, resize_ratios):
    plt.text(x-1, y+0.3, f"{r}", color='blue', ha='center', fontsize=24)
for x, y, r in zip(tflops_mobile_vanilla, acc_mobile_vanilla, resize_ratios):
    plt.text(x+1.5, y-0.7, f"{r}", color='blue', ha='center', fontsize=24)

plt.xlabel("TFLOPs")
plt.ylabel("Accuracy (%)")
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(0, 100)
plt.ylim(86,96)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(20))

plt.savefig("png/ablation_task_mobile.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("pdf/ablation_task_mobile.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
print("📈 Saved at ./png/ablation_task_mobile.png")
print("📈 Saved at ./pdf/ablation_task_mobile.pdf")

# ------------------ Web ------------------
plt.figure(figsize=(10,6))
plt.plot(tflops_web_vanilla, acc_web_vanilla, '^--', linestyle=(0, (3, 3)), color='green', label="Web (GUI-Actor-3B)")
plt.plot(tflops_web_gold, acc_web_gold, 'o-', color='green', label="Web (GUI-Actor-3B + GOLD)")

for x, y, r in zip(tflops_web_gold, acc_web_gold, resize_ratios):
    plt.text(x-1.5, y+0.5, f"{r}", color='green', ha='center', fontsize=24)
for x, y, r in zip(tflops_web_vanilla, acc_web_vanilla, resize_ratios):
    plt.text(x+3.5, y-2, f"{r}", color='green', ha='center', fontsize=24)

plt.xlabel("TFLOPs")
plt.ylabel("Accuracy (%)")
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(0, 100)
plt.ylim(70, 95)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(20))

plt.savefig("png/ablation_task_web.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("pdf/ablation_task_web.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
print("📈 Saved at ./png/ablation_task_web.png")
print("📈 Saved at ./pdf/ablation_task_web.pdf")

# ------------------ Desktop ------------------
plt.figure(figsize=(10,6))
plt.plot(tflops_desktop_vanilla, acc_desktop_vanilla, '^--', linestyle=(0, (3, 3)), color='red', label="Desktop (GUI-Actor-3B)")
plt.plot(tflops_desktop_gold, acc_desktop_gold, 'o-', color='red', label="Desktop (GUI-Actor-3B + GOLD)")

for x, y, r in zip(tflops_desktop_gold, acc_desktop_gold, resize_ratios):
    plt.text(x, y+1.5, f"{r}", color='red', ha='center', fontsize=24)
for x, y, r in zip(tflops_desktop_vanilla, acc_desktop_vanilla, resize_ratios):
    plt.text(x+2, y-3.8, f"{r}", color='red', ha='center', fontsize=24)

plt.xlabel("TFLOPs")
plt.ylabel("Accuracy (%)")
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(0, 40)
plt.ylim(40, 100)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(10))

plt.savefig("png/ablation_task_desktop.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("pdf/ablation_task_desktop.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
print("📈 Saved at ./png/ablation_task_desktop.png")
print("📈 Saved at ./pdf/ablation_task_desktop.pdf")
