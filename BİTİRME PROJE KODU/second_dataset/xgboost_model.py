import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 CSV dosyası kontrol
csv_file = "xgboost_results.csv"
try:
    all_results = pd.read_csv(csv_file)
    if "Model" not in all_results.columns:
        all_results = pd.DataFrame()
except (FileNotFoundError, pd.errors.EmptyDataError):
    all_results = pd.DataFrame()

# 🔹 Parametre kombinasyonları
n_estimators_list = [100, 150, 200, 250]
max_depth_list = [5, 10, 15]
learning_rate_list = [0.01, 0.05, 0.1]

# 🔁 Tüm kombinasyonları dene
for n_est in n_estimators_list:
    for depth in max_depth_list:
        for lr in learning_rate_list:
            if not all_results.empty:
                mask = (
                    (all_results["Model"] == "XGBoost") &
                    (all_results["n_estimators"] == n_est) &
                    (all_results["Max Depth"] == depth) &
                    (all_results["Learning Rate"] == lr)
                )
                if mask.any():
                    continue

            model = XGBClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                eval_metric='logloss',
                random_state=42,
                use_label_encoder=False
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            result = {
                "Model": "XGBoost",
                "n_estimators": n_est,
                "Max Depth": depth,
                "Learning Rate": lr,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0)
            }

            all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)

# 💾 CSV'ye yaz
all_results.to_csv(csv_file, index=False)
print(f"\n📁 Tüm sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🏆 En iyi sonucu yazdır
if not all_results.empty:
    best = all_results[all_results["Model"] == "XGBoost"].sort_values(by="Accuracy", ascending=False).iloc[0]

    print("\n🏆 En iyi XGBoost sonucu:")
    print(f"Model         : {best['Model']}")
    print(f"n_estimators  : {best['n_estimators']}")
    print(f"Max Depth     : {best['Max Depth']}")
    print(f"Learning Rate : {best['Learning Rate']}")
    print(f"Accuracy      : {best['Accuracy']:.4f}")
    print(f"Precision     : {best['Precision']:.4f}")
    print(f"Recall        : {best['Recall']:.4f}")
    print(f"F1 Score      : {best['F1 Score']:.4f}")
else:
    print("⚠️ Sonuçlar boş. Eğitim tamamlanmamış olabilir.")
