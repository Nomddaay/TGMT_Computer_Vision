import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm tính đạo hàm theo x và y


def compute_gradient(image):
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    return magnitude, angle

# Hàm phát hiện điểm đặc trưng


def detect_keypoints(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    magnitude, angle = compute_gradient(gray)

    keypoints = []
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if magnitude[i, j] > threshold:
                # Kiểm tra xem điểm hiện tại có là điểm cực đại trong vùng lân cận không
                if (magnitude[i, j] > magnitude[i-1, j-1] and
                    magnitude[i, j] > magnitude[i-1, j] and
                    magnitude[i, j] > magnitude[i-1, j+1] and
                    magnitude[i, j] > magnitude[i, j-1] and
                    magnitude[i, j] > magnitude[i, j+1] and
                    magnitude[i, j] > magnitude[i+1, j-1] and
                    magnitude[i, j] > magnitude[i+1, j] and
                        magnitude[i, j] > magnitude[i+1, j+1]):
                    keypoints.append((i, j, magnitude[i, j], angle[i, j]))

    return keypoints

# Hàm mô tả điểm đặc trưng


def describe_keypoints(image, keypoints, patch_size=16):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptors = []
    for kp in keypoints:
        i, j, magnitude, angle = kp

        # Tính toán véc-tơ mô tả
        patch = gray[i-patch_size//2:i+patch_size //
                     2, j-patch_size//2:j+patch_size//2]
        hist = cv2.calcHist([patch], [0], None, [8], [0, 256])
        hist /= np.sum(hist)  # Chuẩn hóa histogram

        # Lưu trữ véc-tơ mô tả
        descriptors.append((i, j, magnitude, angle, hist))

    return descriptors


# Đường dẫn đến ảnh
image_path = r'C:\Users\nomdd\OneDrive\Pictures\Screenshots\nam.jpg'

# Đọc ảnh
image = cv2.imread(image_path)

# Thiết lập ngưỡng cho phát hiện điểm đặc trưng
threshold = 50

# Phát hiện điểm đặc trưng
keypoints = detect_keypoints(image, threshold)

# Mô tả điểm đặc trưng
descriptors = describe_keypoints(image, keypoints)

# Hiển thị ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Hiển thị ảnh với các điểm đặc trưng
image_with_keypoints = image.copy()
for kp in keypoints:
    i, j, _, _ = kp
    cv2.circle(image_with_keypoints, (j, i), 3, (0, 255, 0), -1)

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image with Keypoints')
plt.axis('off')

# Hiển thị kết quả
plt.show()

# In ra số lượng điểm đặc trưng được tìm thấy
print(f"Found {len(keypoints)} keypoints.")

# In ra thông tin của các điểm đặc trưng
for kp in descriptors:
    i, j, magnitude, angle, hist = kp
    print(
        f"Keypoint ({i}, {j}): Magnitude={magnitude}, Angle={angle}, Descriptor={hist}")
