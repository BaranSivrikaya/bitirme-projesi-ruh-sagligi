import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 CSV dosyasını yükle veya oluştur
csv_file = "random_forest_results.csv"
try:
    all_results = pd.read_csv(csv_file)
    if "Model" not in all_results.columns:
        all_results = pd.DataFrame()
except (FileNotFoundError, pd.errors.EmptyDataError):
    all_results = pd.DataFrame()

# 🔹 Parametre kombinasyonları
n_estimators_list = [100, 200, 300]
max_depth_list = [10, 20, 30, None]
criterion_list = ["gini", "entropy"]

# 🔁 Kombinasyonları dene
for n_est in n_estimators_list:
    for depth in max_depth_list:
        for crit in criterion_list:
            if not all_results.empty:
                mask = (
                    (all_results["Model"] == "Random Forest") &
                    (all_results["n_estimators"] == n_est) &
                    (all_results["Max Depth"].astype(str) == str(depth)) &
                    (all_results["Criterion"] == crit)
                )
                if mask.any():
                    continue

            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                criterion=crit,
                random_state=42
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            result = {
                "Model": "Random Forest",
                "n_estimators": n_est,
                "Max Depth": depth,
                "Criterion": crit,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0)
            }

            result_df = pd.DataFrame([result])

            # 🔧 Boş veya tümü NA olmayan sonuçları sadece ekle
            if not result_df.empty and not result_df.isna().all(axis=None):
                if all_results.empty:
                    all_results = result_df.copy()
                else:
                    all_results = pd.concat([all_results, result_df], ignore_index=True)

# 💾 CSV'ye yaz
all_results.to_csv(csv_file, index=False)
print(f"📁 Tüm sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🏆 En iyi sonucu yazdır
if not all_results.empty:
    best = all_results[all_results["Model"] == "Random Forest"].sort_values(by="Accuracy", ascending=False).iloc[0]

    print("\n🏆 En iyi Random Forest sonucu:")
    print(f"Model        : {best['Model']}")
    print(f"n_estimators : {best['n_estimators']}")
    print(f"Max Depth    : {best['Max Depth']}")
    print(f"Criterion    : {best['Criterion']}")
    print(f"Accuracy     : {best['Accuracy']:.4f}")
    print(f"Precision    : {best['Precision']:.4f}")
    print(f"Recall       : {best['Recall']:.4f}")
    print(f"F1 Score     : {best['F1 Score']:.4f}")
else:
    print("⚠️ Sonuçlar boş. Eğitim tamamlanmamış olabilir.")
