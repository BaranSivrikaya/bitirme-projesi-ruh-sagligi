import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 Denenecek var_smoothing değerleri
smoothing_values = [1e-9, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4]

# 🔹 CSV dosyasını yükle veya oluştur
csv_file = "naive_bayes_results.csv"
try:
    all_results = pd.read_csv(csv_file)
    if "Model" not in all_results.columns:
        all_results = pd.DataFrame()
except (FileNotFoundError, pd.errors.EmptyDataError):
    all_results = pd.DataFrame()

# 🔁 Her smoothing değerini sırayla dene
for smooth in smoothing_values:
    if not all_results.empty:
        mask = (
            (all_results["Model"] == "Naive Bayes") &
            (all_results["var_smoothing"] == smooth)
        )
        if mask.any():
            continue

    model = GaussianNB(var_smoothing=smooth)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    result = {
        "Model": "Naive Bayes",
        "var_smoothing": smooth,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0)
    }

    all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)

# 🔹 CSV'ye kaydet
all_results.to_csv(csv_file, index=False)
print(f"📁 Tüm sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🏆 En iyi sonucu yazdır
if not all_results.empty:
    best = all_results[all_results["Model"] == "Naive Bayes"].sort_values(by="Accuracy", ascending=False).iloc[0]

    print("\n🏆 En iyi Naive Bayes sonucu:")
    print(f"Model          : {best['Model']}")
    print(f"var_smoothing  : {best['var_smoothing']}")
    print(f"Accuracy       : {best['Accuracy']:.4f}")
    print(f"Precision      : {best['Precision']:.4f}")
    print(f"Recall         : {best['Recall']:.4f}")
    print(f"F1 Score       : {best['F1 Score']:.4f}")
else:
    print("⚠️ Henüz sonuç yok veya CSV tamamen boş.")
