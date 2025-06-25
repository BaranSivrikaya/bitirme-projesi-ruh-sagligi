import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 Parametreler
criterions = ["gini", "entropy"]
max_depths = [None, 5, 10, 15, 20, 30, 50]

# 🔹 CSV dosyasını yükle (varsa), yoksa boş başlat
csv_file = "decision_tree_results.csv"
try:
    all_results = pd.read_csv(csv_file)
    # Gerekli kolonlar yoksa boş DataFrame'e döndür
    if "Model" not in all_results.columns:
        all_results = pd.DataFrame()
except (FileNotFoundError, pd.errors.EmptyDataError):
    all_results = pd.DataFrame()

# 🔁 Kombinasyonları dene
for criterion in criterions:
    for max_depth in max_depths:
        if not all_results.empty:
            mask = (
                (all_results["Model"] == "Decision Tree") &
                (all_results["Criterion"] == criterion) &
                (all_results["Max Depth"].astype(str) == str(max_depth))
            )
            if mask.any():
                continue

        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        result = {
            "Model": "Decision Tree",
            "Criterion": criterion,
            "Max Depth": max_depth,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0)
        }

        all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)

# 🔹 CSV'ye kaydet
all_results.to_csv(csv_file, index=False)
print(f"📁 Tüm sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🔍 En iyi sonucu terminale yazdır
if not all_results.empty:
    best = all_results[all_results["Model"] == "Decision Tree"].sort_values(by="Accuracy", ascending=False).iloc[0]

    print("\n🏆 En iyi Decision Tree sonucu:")
    print(f"Model         : {best['Model']}")
    print(f"Criterion     : {best['Criterion']}")
    print(f"Max Depth     : {best['Max Depth']}")
    print(f"Accuracy      : {best['Accuracy']:.4f}")
    print(f"Precision     : {best['Precision']:.4f}")
    print(f"Recall        : {best['Recall']:.4f}")
    print(f"F1 Score      : {best['F1 Score']:.4f}")
else:
    print("⚠️ Hiçbir sonuç bulunamadı veya CSV dosyası boş.")
