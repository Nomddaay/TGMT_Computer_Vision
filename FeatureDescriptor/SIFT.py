import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

image_path = r"C:\Users\nomdd\OneDrive\Pictures\Screenshots\cay.jpg"

image = Image.open(image_path)

image_gray = image.convert("L")

if image_gray is None:
    print("Không thể đọc được ảnh", image_path)
else:
    def compute_gradients(image):


        # Lọc Gauss để làm mờ ảnh
        blurred = gaussian_filter(image, sigma=1.5)

        # Tính toán gradient theo trục x và trục y
        gradient_x = np.gradient(blurred, axis=1)
        gradient_y = np.gradient(blurred, axis=0)

        # Tính toán độ lớn gradient và hướng gradient
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_magnitude, gradient_direction


    def sift(image):
        # Bước 1: Xác định điểm đặc trưng (Keypoint detection)
        # Xác định độ lớn và hướng gradient
        gradient_magnitude, gradient_direction = compute_gradients(image)

        # Vẽ các keypoint lên ảnh
        keypoint_coordinates = np.argwhere(gradient_magnitude > 15)  # Chỉ vẽ các keypoint có độ lớn lớn hơn ngưỡng 100 (điều chỉnh ngưỡng tùy ý)
        plt.imshow(image, cmap='gray')
        plt.scatter(keypoint_coordinates[:, 1], keypoint_coordinates[:, 0], color='red', s=5)  # Vẽ các keypoint bằng điểm màu đỏ
        plt.axis('off')
        plt.show()


    sift(image_gray)
