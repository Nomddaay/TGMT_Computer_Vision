import cv2
from skimage.feature import hog

# Đọc hình ảnh
image = cv2.imread('images/random_images/mck.jpg')

# Chuyển đổi hình ảnh sang grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Rút trích đặc trưng HOG từ hình ảnh grayscale
features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, transform_sqrt=True)

# In ra kích thước của vectơ đặc trưng
print("Kich thuoc dac trung HOG:", features.shape)
