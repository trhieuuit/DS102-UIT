from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Chạy được từ thư mục repo hoặc từ LAB02.
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if not (ROOT / "LAB01").exists():
    ROOT = ROOT.parent

df = pd.read_csv(ROOT / "LAB01/Bai02/FPT_stock.csv", parse_dates=["date"])
df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
print("Number of sessions:", len(df))
print(df.head().to_string(index=False))

data = pd.DataFrame({"date": df["date"]})
data["return_today"] = df["adjusted_close"].pct_change(fill_method=None)
data["intraday_return"] = df["close"] / df["open"] - 1
data["price_range"] = (df["high"] - df["low"]) / df["open"]
data["volume_change"] = df["volume"].pct_change(fill_method=None)
data["ma5_ratio"] = df["adjusted_close"] / df["adjusted_close"].rolling(5).mean() - 1
data["next_return"] = df["adjusted_close"].shift(-1) / df["adjusted_close"] - 1
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

features = ["return_today", "intraday_return", "price_range", "volume_change", "ma5_ratio"]
X = data[features].to_numpy()
y_binary = (data["next_return"] > 0).astype(int).to_numpy()
y_multi = np.where(data["next_return"] < -0.01, 0,
                   np.where(data["next_return"] > 0.01, 2, 1))

# 80% phiên đầu để train, 20% phiên cuối để test. 
split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
yb_train, yb_test = y_binary[:split], y_binary[split:]
ym_train, ym_test = y_multi[:split], y_multi[split:]

# Chỉ tính trung bình và độ lệch chuẩn trên tập train để tránh rò rỉ dữ liệu.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Thêm cột 1 để hệ số đầu tiên đóng vai trò bias trong mô hình NumPy.
X_train_bias = np.column_stack([np.ones(len(X_train)), X_train])
X_test_bias = np.column_stack([np.ones(len(X_test)), X_test])

print("Train:", len(X_train), "| Test:", len(X_test))
print("Training feature dates:", data["date"].iloc[0].date(), "to", data["date"].iloc[split - 1].date())
print("Test feature dates:", data["date"].iloc[split].date(), "to", data["date"].iloc[-1].date())
print("Class counts (train/test):")
print("Binary:", np.bincount(yb_train, minlength=2), np.bincount(yb_test, minlength=2))
print("Three classes:", np.bincount(ym_train, minlength=3), np.bincount(ym_test, minlength=3))

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def train_logistic(X, y, learning_rate=0.1, epochs=2000):
    w = np.zeros(X.shape[1])
    losses = []
    for epoch in range(epochs):
        z = X @ w
        p = sigmoid(z)
        # logaddexp tính binary cross-entropy ổn định, tránh log(0).
        loss = np.mean(np.logaddexp(0, z) - y * z)
        losses.append(loss)
        gradient = X.T @ (p - y) / len(y)
        w -= learning_rate * gradient
    return w, losses


w, logistic_losses = train_logistic(X_train_bias, yb_train)
pred_binary = (sigmoid(X_test_bias @ w) >= 0.5).astype(int)
print(classification_report(yb_test, pred_binary, labels=[0, 1],
                            target_names=["Non-Increase", "Increase"], digits=4, zero_division=0))

plt.figure(figsize=(7, 4))
plt.plot(logistic_losses)
plt.title("Logistic Regression - Training loss")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def train_softmax(X, y, learning_rate=0.1, epochs=2000):
    W = np.zeros((X.shape[1], 3))
    Y = np.eye(3)[y]
    losses = []
    for epoch in range(epochs):
        P = softmax(X @ W)
        loss = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1)))
        losses.append(loss)
        gradient = X.T @ (P - Y) / len(y)
        W -= learning_rate * gradient
    return W, losses


W, softmax_losses = train_softmax(X_train_bias, ym_train)
pred_multi = softmax(X_test_bias @ W).argmax(axis=1)
print(classification_report(ym_test, pred_multi, labels=[0, 1, 2],
                            target_names=["Decrease", "Sideways", "Increase"], digits=4, zero_division=0))

plt.figure(figsize=(7, 4))
plt.plot(softmax_losses)
plt.title("Softmax Regression - Training loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

logistic_sklearn = LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000)
logistic_sklearn.fit(X_train, yb_train)
pred_binary_sk = logistic_sklearn.predict(X_test)
print("Logistic Regression - Scikit-learn")
print(classification_report(yb_test, pred_binary_sk, labels=[0, 1],
                            target_names=["Non-Increase", "Increase"], digits=4, zero_division=0))

softmax_sklearn = LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000)
softmax_sklearn.fit(X_train, ym_train)
pred_multi_sk = softmax_sklearn.predict(X_test)
print("Softmax Regression - Scikit-learn")
print(classification_report(ym_test, pred_multi_sk, labels=[0, 1, 2],
                            target_names=["Decrease", "Sideways", "Increase"], digits=4, zero_division=0))

from sklearn.metrics import precision_recall_fscore_support

baseline_binary = np.full(len(yb_test), np.bincount(yb_train).argmax())
baseline_multi = np.full(len(ym_test), np.bincount(ym_train).argmax())
results = []
for name, actual, predicted, labels in [
    ("Logistic NumPy", yb_test, pred_binary, [0, 1]),
    ("Logistic sklearn", yb_test, pred_binary_sk, [0, 1]),
    ("Binary baseline", yb_test, baseline_binary, [0, 1]),
    ("Softmax NumPy", ym_test, pred_multi, [0, 1, 2]),
    ("Softmax sklearn", ym_test, pred_multi_sk, [0, 1, 2]),
    ("Three-class baseline", ym_test, baseline_multi, [0, 1, 2]),
]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, average="macro", zero_division=0)
    results.append([name, accuracy_score(actual, predicted), precision, recall, f1])

summary = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision macro", "Recall macro", "F1 macro"])
print(summary.round(4).to_string(index=False))
print("Logistic loss: %.4f -> %.4f" % (logistic_losses[0], logistic_losses[-1]))
print("Softmax loss: %.4f -> %.4f" % (softmax_losses[0], softmax_losses[-1]))
