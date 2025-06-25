import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 Denenecek parametre kombinasyonları
n_estimators_list = [100, 150, 200]
max_depth_list = [5, 10, 15, 20]
learning_rate_list = [0.01, 0.05, 0.1]

# 🔹 CSV yükle veya oluştur
csv_file = "lightgbm_results.csv"
if os.path.exists(csv_file):
    all_results = pd.read_csv(csv_file)
else:
    all_results = pd.DataFrame()

# 🔁 Tüm kombinasyonları dene
for n_est in n_estimators_list:
    for depth in max_depth_list:
        for lr in learning_rate_list:
            mask = (
                (all_results["Model"] == "LightGBM") &
                (all_results["n_estimators"] == n_est) &
                (all_results["Max Depth"] == depth) &
                (all_results["Learning Rate"] == lr)
            )
            if not all_results.empty and mask.any():
                continue

            model = LGBMClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                random_state=42
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            result = {
                "Model": "LightGBM",
                "n_estimators": n_est,
                "Max Depth": depth,
                "Learning Rate": lr,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0)
            }

            all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)

# 🔹 CSV'ye yaz
all_results.to_csv(csv_file, index=False)
print(f"📁 Tüm sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🏆 En iyi sonucu yazdır
best = all_results[all_results["Model"] == "LightGBM"].sort_values(by="Accuracy", ascending=False).iloc[0]

print("\n🏆 En iyi LightGBM sonucu:")
print(f"Model         : {best['Model']}")
print(f"n_estimators  : {best['n_estimators']}")
print(f"Max Depth     : {best['Max Depth']}")
print(f"Learning Rate : {best['Learning Rate']}")
print(f"Accuracy      : {best['Accuracy']:.4f}")
print(f"Precision     : {best['Precision']:.4f}")
print(f"Recall        : {best['Recall']:.4f}")
print(f"F1 Score      : {best['F1 Score']:.4f}")
