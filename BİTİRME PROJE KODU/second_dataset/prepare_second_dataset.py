import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 🔹 Veri setini yükle
df = pd.read_csv("survey.csv")

# 🔹 Gereksiz sütunları kaldır (isteğe göre değişebilir)
df = df.drop(columns=["Timestamp", "state", "comments"], errors="ignore")

# 🔹 Eksik verileri "Unknown" ile doldur
df = df.fillna("Unknown")

# 🔹 Kategorik sütunları sayısallaştır
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 🔹 X ve y'yi ayır
X = df.drop(columns=["treatment"])
y = df["treatment"]

# 🔹 Eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 CSV olarak kaydet
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Veriler başarıyla hazırlandı ve kaydedildi.")
