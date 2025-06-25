import pandas as pd
import matplotlib.pyplot as plt

# En iyi sonuçlar (Accuracy'e göre seçilmiş) - İkinci veri seti için
data = {
    "Model": [
        "Decision Tree", "Extra Trees", "KNN", "LightGBM",
        "Logistic Regression", "Naive Bayes", "Neural Net",
        "Random Forest", "XGBoost"
    ],
    "Accuracy": [
        0.8056, 0.8135, 0.6865, 0.8135, 0.7341, 0.5119, 0.7738, 0.8254, 0.8175
    ],
    "Precision": [
        0.7372, 0.7923, 0.6746, 0.7836, 0.7333, 0.0000, 0.7578, 0.7970, 0.7852
    ],
    "Recall": [
        0.9350, 0.8374, 0.6911, 0.8537, 0.7154, 0.0000, 0.7886, 0.8618, 0.8618
    ],
    "F1 Score": [
        0.8244, 0.8142, 0.6827, 0.8171, 0.7243, 0.0000, 0.7729, 0.8281, 0.8218
    ]
}

# DataFrame'e dönüştür
df = pd.DataFrame(data)

# Grafik çizimi
plt.figure(figsize=(14, 6))
plt.plot(df["Model"], df["Accuracy"], marker='o', label="Accuracy")
plt.plot(df["Model"], df["Precision"], marker='o', label="Precision")
plt.plot(df["Model"], df["Recall"], marker='o', label="Recall")
plt.plot(df["Model"], df["F1 Score"], marker='o', label="F1 Score")

plt.title("İkinci Veri Seti - Makine Öğrenmesi Modellerinin Performans Karşılaştırması")
plt.xlabel("Model")
plt.ylabel("Değer")
plt.xticks(rotation=30)
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
