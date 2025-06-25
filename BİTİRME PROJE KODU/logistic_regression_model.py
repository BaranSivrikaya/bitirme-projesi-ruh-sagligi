import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 Parametre kombinasyonları
penalty = "l2"
C_list = [0.1, 1.0, 10.0, 100.0]
solver_list = ["liblinear", "lbfgs"]

# 🔹 CSV dosyası kontrolü
csv_file = "logistic_regression_results.csv"
if os.path.exists(csv_file):
    all_results = pd.read_csv(csv_file)
else:
    all_results = pd.DataFrame()

# 🔁 Kombinasyonları sırayla dene
for C_val in C_list:
    for solver in solver_list:
        mask = (
            (all_results["Model"] == "Logistic Regression") &
            (all_results["Penalty"] == penalty) &
            (all_results["C"] == C_val) &
            (all_results["Solver"] == solver)
        )
        if not all_results.empty and mask.any():
            continue

        try:
            model = LogisticRegression(
                penalty=penalty,
                C=C_val,
                solver=solver,
                max_iter=1000,
                random_state=42
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            result = {
                "Model": "Logistic Regression",
                "Penalty": penalty,
                "C": C_val,
                "Solver": solver,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0)
            }

            all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)

        except Exception as e:
            print(f"⚠️ Hata oluştu (C={C_val}, Solver={solver}): {e}")

# 🔹 CSV'ye yaz
all_results.to_csv(csv_file, index=False)
print(f"📁 Tüm sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🏆 En iyi sonucu yazdır
best = all_results[all_results["Model"] == "Logistic Regression"].sort_values(by="Accuracy", ascending=False).iloc[0]

print("\n🏆 En iyi Logistic Regression sonucu:")
print(f"Model     : {best['Model']}")
print(f"Penalty   : {best['Penalty']}")
print(f"C         : {best['C']}")
print(f"Solver    : {best['Solver']}")
print(f"Accuracy  : {best['Accuracy']:.4f}")
print(f"Precision : {best['Precision']:.4f}")
print(f"Recall    : {best['Recall']:.4f}")
print(f"F1 Score  : {best['F1 Score']:.4f}")
