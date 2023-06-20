import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# Chuẩn bị dữ liệu
digits = load_digits()
X = digits.data
y = digits.target

# Chuẩn hoá dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Xây dựng mô hình k-NN
k = 3  # Số láng giềng gần nhất
knn = KNeighborsClassifier(n_neighbors=k)

# Huấn luyện mô hình
knn.fit(X_train, y_train)

# Dự đoán nhãn cho tập kiểm tra
y_pred = knn.predict(X_test)

# Đánh giá độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
