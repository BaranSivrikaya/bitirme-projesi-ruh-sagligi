import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 Parametre kombinasyonları
n_estimators_list = [100, 150, 200]
max_depth_list = [None, 10, 20, 30, 40]
criterion_list = ["gini", "entropy"]

# 🔹 CSV dosyasını yükle (varsa), yoksa boş oluştur
csv_file = "extra_trees_results.csv"
if os.path.exists(csv_file):
    all_results = pd.read_csv(csv_file)
else:
    all_results = pd.DataFrame()

# 🔁 Tüm kombinasyonları sırayla uygula
for n_est in n_estimators_list:
    for depth in max_depth_list:
        for crit in criterion_list:

            # Bu kombinasyon daha önce işlendi mi?
            mask = (
                (all_results["Model"] == "Extra Trees") &
                (all_results["n_estimators"] == n_est) &
                (all_results["Max Depth"].astype(str) == str(depth)) &
                (all_results["Criterion"] == crit)
            )

            if not all_results.empty and mask.any():
                continue

            # 🔨 Modeli kur ve eğit
            model = ExtraTreesClassifier(
                n_estimators=n_est,
                max_depth=depth,
                criterion=crit,
                random_state=42
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            # 🔹 Metrikler
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, zero_division=0)
            recall = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)

            result = {
                "Model": "Extra Trees",
                "n_estimators": n_est,
                "Max Depth": depth,
                "Criterion": crit,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }

            result_df = pd.DataFrame([result])
            if not all_results.empty:
                all_results = pd.concat([all_results, result_df], ignore_index=True)
            else:
                all_results = result_df

# 🔹 CSV'ye kaydet
all_results.to_csv(csv_file, index=False)
print(f"📁 Sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🔍 En iyi sonucu göster
best = all_results[all_results["Model"] == "Extra Trees"].sort_values(by="Accuracy", ascending=False).iloc[0]

print("\n🏆 En iyi Extra Trees sonucu:")
print(f"Model         : {best['Model']}")
print(f"n_estimators  : {best['n_estimators']}")
print(f"Max Depth     : {best['Max Depth']}")
print(f"Criterion     : {best['Criterion']}")
print(f"Accuracy      : {best['Accuracy']:.4f}")
print(f"Precision     : {best['Precision']:.4f}")
print(f"Recall        : {best['Recall']:.4f}")
print(f"F1 Score      : {best['F1 Score']:.4f}")
