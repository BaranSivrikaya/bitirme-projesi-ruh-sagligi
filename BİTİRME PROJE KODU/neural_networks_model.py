import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os

# 🔹 Verileri oku
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔹 Verileri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 CSV kontrol
csv_file = "neural_networks_results.csv"
if os.path.exists(csv_file):
    all_results = pd.read_csv(csv_file)
else:
    all_results = pd.DataFrame()

# 🔹 Denenecek kombinasyonlar
hidden_layers = [(50,), (100,), (100, 50)]
activations = ['relu', 'tanh']
solvers = ['adam', 'sgd']

# 🔁 Tüm kombinasyonları dene
for h in hidden_layers:
    for act in activations:
        for solver in solvers:
            mask = (
                (all_results["Model"] == "Neural Network") &
                (all_results["hidden_layers"] == str(h)) &
                (all_results["activation"] == act) &
                (all_results["solver"] == solver)
            )
            if not all_results.empty and mask.any():
                continue

            model = MLPClassifier(
                hidden_layer_sizes=h,
                activation=act,
                solver=solver,
                max_iter=500,
                random_state=42
            )

            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            result = {
                "Model": "Neural Network",
                "hidden_layers": str(h),
                "activation": act,
                "solver": solver,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, zero_division=0),
                "Recall": recall_score(y_test, preds, zero_division=0),
                "F1 Score": f1_score(y_test, preds, zero_division=0)
            }

            all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)

# 💾 Sonuçları CSV’ye kaydet
all_results.to_csv(csv_file, index=False)
print(f"📁 Tüm sonuçlar CSV dosyasına kaydedildi: {csv_file}")

# 🏆 En iyi sonucu terminale yazdır
best = all_results[all_results["Model"] == "Neural Network"].sort_values(by="Accuracy", ascending=False).iloc[0]

print("\n🏆 En iyi Neural Network sonucu:")
print(f"Model         : {best['Model']}")
print(f"hidden_layers : {best['hidden_layers']}")
print(f"activation    : {best['activation']}")
print(f"solver        : {best['solver']}")
print(f"Accuracy      : {best['Accuracy']:.4f}")
print(f"Precision     : {best['Precision']:.4f}")
print(f"Recall        : {best['Recall']:.4f}")
print(f"F1 Score      : {best['F1 Score']:.4f}")
