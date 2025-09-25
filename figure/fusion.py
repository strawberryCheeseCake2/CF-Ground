import matplotlib.pyplot as plt

# 전역 폰트 설정
plt.rc('font', size=28)
plt.rc('axes', titlesize=28, labelsize=28)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)
plt.rc('legend', fontsize=22)
textfontsize = 24

alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
acc = [89.62, 89.78, 89.86, 89.86, 90.33, 90.57, 91.04, 91.43, 90.96, 90.57, 90.49]

plt.figure(figsize=(10,6))
plt.plot(alpha, acc, 'o-', color='orange', label="Accuracy")

for x, y in zip(alpha, acc):
    plt.text(x, y+0.2, f"{x}", color='orange', ha='center', fontsize=textfontsize)  # 크기 조정

plt.xlabel("Fusion Ratio α")
plt.ylabel("Accuracy (%)")
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(88, 93)

plt.savefig("png/ablation_fusion_ratio.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("pdf/ablation_fusion_ratio.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

print("📈 Saved at ./png/ablation_fusion_ratio.png")
print("📈 Saved at ./pdf/ablation_fusion_ratio.pdf")
