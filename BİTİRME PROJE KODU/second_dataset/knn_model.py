import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 Parametre kombinasyonları
n_neighbors_list = [3, 5, 7, 9, 11, 13, 15]
metric_list = ["euclidean", "manhattan"]

# 🔹 CSV dosyasını yükle (varsa), yoksa boş oluştur
csv_file = "knn_results.csv"
try:
    all_results = pd.read_csv(csv_file)
    if "Model" not in all_results.columns:
        all_results = pd.DataFrame()
except (FileNotFoundError, pd.errors.EmptyDataError):
    all_results = pd.DataFrame()

# 🔁 Kombinasyonları dene
for n_neigh in n_neighbors_list:
    for metric in metric_list:
        if not all_results.empty:
            mask = (
                (all_results["Model"] == "K-Nearest Neighbors") &
                (all_results["n_neighbors"] == n_neigh) &
                (all_results["Metric"] == metric)
            )
            if mask.any():
                continue

        model = KNeighborsClassifier(n_neighbors=n_neigh, metric=metric)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        result = {
            "Model": "K-Nearest Neighbors",
            "n_neighbors": n_neigh,
            "Metric": metric,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)

# 🔹 CSV'ye kaydet
all_results.to_csv(csv_file, index=False)
print(f"📁 Sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🔍 En iyi sonucu terminale yazdır
if not all_results.empty:
    best = all_results[all_results["Model"] == "K-Nearest Neighbors"].sort_values(by="Accuracy", ascending=False).iloc[0]

    print("\n🏆 En iyi K-Nearest Neighbors sonucu:")
    print(f"Model       : {best['Model']}")
    print(f"n_neighbors : {best['n_neighbors']}")
    print(f"Metric      : {best['Metric']}")
    print(f"Accuracy    : {best['Accuracy']:.4f}")
    print(f"Precision   : {best['Precision']:.4f}")
    print(f"Recall      : {best['Recall']:.4f}")
    print(f"F1 Score    : {best['F1 Score']:.4f}")
else:
    print("⚠️ Henüz sonuç bulunamadı. CSV boş olabilir.")
