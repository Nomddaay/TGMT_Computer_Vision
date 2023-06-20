import os
import numpy as np
from PIL import Image, ImageOps

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            distances = self.calculate_distances(X_test[i])
            k_nearest_labels = [self.y_train[idx] for idx in distances.argsort()[:self.k]]
            y_pred.append(self.most_common_label(k_nearest_labels))
        return y_pred

    def calculate_distances(self, x):
        distances = []

        for i in range(len(self.X_train)):
            dist = np.sqrt(np.sum(np.square(x - self.X_train[i])))
            distances.append(dist)
        return np.array(distances)

    def most_common_label(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count_idx = np.argmax(counts)
        return unique_labels[max_count_idx]

# Chuyển đổi hình ảnh thành vector đặc trưng
def extract_features(image):
    # Chuyển đổi hình ảnh thành mảng numpy
    image_array = np.array(image)

    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    normalized_image = image_array / 255.0

    # Chuyển đổi ma trận 3D (chiều rộng, chiều cao, kênh màu) thành vector 1D
    print(normalized_image.shape)
    features = normalized_image.flatten()

    return features

# Đường dẫn tới thư mục chứa các hình ảnh huấn luyện
train_folder = ""

# Danh sách các hình ảnh huấn luyện và nhãn tương ứng
train_images = []
for filename in os.listdir(train_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(train_folder, filename)
        label = filename.split(".")[0]  # Giả sử tên file chứa nhãn
        train_images.append((image_path, label))

# Chuyển đổi hình ảnh huấn luyện thành các vector đặc trưng và nhãn
X_train = []
y_train = []
target_size = (300, 300)
for image_path, label in train_images:
    image = Image.open(image_path)
    gray_image = ImageOps.grayscale(image)
    gray_image = gray_image.resize(target_size)
    features = extract_features(gray_image)
    X_train.append(features)
    y_train.append(label)