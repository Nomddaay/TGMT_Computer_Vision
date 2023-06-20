import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def compute_gradients(image):
    # Chuyển đổi hình ảnh sang ảnh xám nếu cần thiết
    if len(image.shape) > 2:
        gray_image = np.mean(image, axis=2) #tính giá trị trung bình của các kênh màu trong image theo trục thứ ba (axis=2)
    else:
        gray_image = image

    # Áp dụng bộ lọc Sobel để tính toán gradient theo hướng x và y
    gradient_x = ndimage.sobel(gray_image, axis=1)
    gradient_y = ndimage.sobel(gray_image, axis=0)

    # Tính toán độ lớn và hướng gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

    return magnitude, orientation
image = plt.imread(r'C:\Users\nomdd\OneDrive\Pictures\Screenshots\ava.jpg')

# Tính toán gradient
magnitude, orientation = compute_gradients(image)

# Hiển thị hình ảnh
plt.figure()
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')

# Hiển thị gradient theo độ lớn
plt.figure()
plt.imshow(magnitude, cmap='hot')
plt.axis('off')
plt.title('Gradient Magnitude')

# Hiển thị gradient theo hướng
plt.figure()
plt.imshow(orientation, cmap='hsv')
plt.axis('off')
plt.title('Gradient Orientation')

plt.show()