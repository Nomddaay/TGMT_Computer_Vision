import cv2
import numpy as np

def compute_gradients(image):
    # Chuyển đổi hình ảnh sang ảnh xám nếu cần thiết
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Áp dụng bộ lọc Sobel để tính toán gradient theo hướng x và y
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Tính toán độ lớn và hướng gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

    return magnitude, orientation

def divide_image_into_cells(image, cell_size, cell_stride):
    height, width = image.shape[:2]
    cells = []

    for y in range(0, height - cell_size + 1, cell_stride):
        for x in range(0, width - cell_size + 1, cell_stride):
            cell = image[y:y+cell_size, x:x+cell_size]
            cells.append(cell)

    return cells

def compute_histograms(cells, orientation, num_bins):
    histograms = []

    for cell, cell_orientation in zip(cells, orientation):
        # Tạo histogram với số lượng bin đã định trước
        histogram = np.zeros(num_bins)

        # Tính toán và tăng giá trị của bin tương ứng cho mỗi gradient trong ô lưới
        for i in range(cell.shape[0]):
            for j in range(cell.shape[1]):
                bin_index = int(cell_orientation[i] / (360 / num_bins))
                histogram[bin_index] += 1

        histograms.append(histogram)

    return histograms

def normalize_blocks(histograms, block_size):
    normalized_blocks = []

    for i in range(len(histograms) - block_size + 1):
        block = np.concatenate(histograms[i:i+block_size])
        block /= np.linalg.norm(block) + 1e-5  # Chuẩn hóa biểu đồ

        normalized_blocks.append(block)

    return normalized_blocks

def combine_histograms(normalized_blocks):
    feature_vector = np.concatenate(normalized_blocks)

    return feature_vector

def apply_l2_normalization(feature_vector):
    norm = np.linalg.norm(feature_vector)
    normalized_vector = feature_vector / (norm + 1e-5)  # Chuẩn hóa L2

    return normalized_vector

# Đọc hình ảnh
image = cv2.imread(r'C:\Users\nomdd\OneDrive\Pictures\Screenshots\cay.jpg')

# Tính toán gradient
magnitude, orientation = compute_gradients(image)

# Chia hình ảnh thành các ô lưới
cell_size = 8  # Kích thước ô lưới
cell_stride = 8  # Bước nhảy giữa các ô lưới
cells = divide_image_into_cells(magnitude, cell_size, cell_stride)

# Tính toán histogram
num_bins = 9  # Số lượng bin trong histogram
histograms = compute_histograms(cells, orientation, num_bins)

# In ra các histogram
for i, histogram in enumerate(histograms):
    print("Histogram {}: {}".format(i, histogram))

# Chuẩn hóa các khối
block_size = 2  # Kích thước khối
normalized_blocks = normalize_blocks(histograms, block_size)

# In ra kích thước và giá trị của các khối
for i, block in enumerate(normalized_blocks):
    print("Block {}: Shape: {}, Values: {}".format(i, block.shape, block))

# Kết hợp các biểu đồ độ lệch hướng thành một vector đặc trưng duy nhất
feature_vector = combine_histograms(normalized_blocks)

# Áp dụng chuẩn hóa L2 cho vector đặc trưng
normalized_vector = apply_l2_normalization(feature_vector)

print("Feature Vector:", normalized_vector)

# Hiển thị các ô lưới
cv2.imshow("Magnitude", magnitude.astype(np.uint8))
cv2.imshow("Orientation", orientation.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

