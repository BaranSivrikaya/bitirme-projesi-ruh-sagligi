import matplotlib.pyplot as plt
import pandas as pd

# 🔹 En iyi sonuçlardan oluşan veriler
data = {
    "Model": [
        "Decision Tree", "Extra Trees", "KNN", "LightGBM",
        "Logistic Regression", "Naive Bayes", "Neural Net",
        "Random Forest", "XGBoost"
    ],
    "Accuracy": [
        0.9977, 0.8845, 0.9935, 0.9922,
        0.7021, 0.6966, 0.9771,
        0.9189, 0.9944
    ],
    "Precision": [
        0.9976, 0.8896, 0.9968, 0.9920,
        0.7024, 0.7104, 0.9713,
        0.9303, 0.9953
    ],
    "Recall": [
        0.9977, 0.8811, 0.9903, 0.9926,
        0.7133, 0.6756, 0.9838,
        0.9074, 0.9943
    ],
    "F1 Score": [
        0.9977, 0.8853, 0.9935, 0.9923,
        0.7078, 0.6926, 0.9775,
        0.9187, 0.9944
    ]
}

df = pd.DataFrame(data)

# 🔸 Grafik çizimi
plt.figure(figsize=(14, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
colors = ["orange", "darkorange", "deeppink", "magenta"]

for metric, color in zip(metrics, colors):
    plt.plot(df["Model"], df[metric], marker='o', label=metric, color=color)

# 🔸 Görsel ayarları
plt.title("Makine Öğrenmesi Modellerinin Performans Karşılaştırması", fontsize=14)
plt.xlabel("Model")
plt.ylabel("Değer")
plt.xticks(rotation=30)
plt.ylim(0.65, 1.01)
plt.legend()
plt.tight_layout()

# 🔸 Kaydet ve göster
plt.savefig("model_performance_comparison.png", dpi=300)
plt.show()
