import numpy as np
from PIL import Image

def get_lbp_pixel(image, x, y):
    center_value = image[x, y]  # Giá trị pixel tại điểm ảnh trung tâm

    lbp_code = 0

    # Xác định vùng lân cận xung quanh
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    # Xác định giá trị cho các điểm xung quanh trong vùng lân cận
    for i, j in neighbors:
        new_x = x + i
        new_y = y + j

        # Kiểm tra xem vị trí mới có nằm trong kích thước ảnh hay không
        if 0 <= new_x < image.shape[0] and 0 <= new_y < image.shape[1]:
            if image[new_x, new_y] >= center_value:
                lbp_code |= (1 << (7 - (neighbors.index((i, j)) + 1) % 8))  # Đặt bit 1
            else:
                lbp_code |= (0 << (7 - (neighbors.index((i, j)) + 1) % 8))  # Đặt bit 0

    return lbp_code

def lbp_descriptor(image):
    height, width = image.shape

    # Tạo ma trận LBP với cùng kích thước ảnh đầu vào
    lbp_image = np.zeros((height, width), dtype=np.uint8)

    # Lặp qua từng điểm ảnh trong ảnh
    for i in range(height):
        for j in range(width):
            lbp_code = get_lbp_pixel(image, i, j)  # Bước 2: Xác định giá trị LBP cho điểm ảnh
            lbp_image[i, j] = lbp_code

    return lbp_image

def lbp_to_vector(lbp_code):
    binary_str = bin(lbp_code)[2:].zfill(8)  # Chuyển đổi LBP code sang chuỗi nhị phân 8-bit
    binary_vector = [int(bit) for bit in binary_str]  # Chuyển đổi chuỗi nhị phân thành vector đặc trưng
    return binary_vector

# Đường dẫn tới tập tin ảnh trên máy tính
image_path = r'C:\Users\nomdd\OneDrive\Pictures\Screenshots\mck.jpg'

# Đọc ảnh và chuyển đổi thành ảnh xám
image = Image.open(image_path).convert("L")

# Chuyển đổi ảnh thành mảng numpy
image_array = np.array(image)

# Tính toán ma trận LBP descriptor
lbp_result = lbp_descriptor(image_array)

#Tạo chuỗi nhị phân từ ma trận LBP
binary_vector = [lbp_to_vector(pixel) for row in lbp_result for pixel in row]

x = 100
y = 150

lbp_value = get_lbp_pixel(image_array, x, y)
print("LBP Descriptor:")
print(lbp_result)
print("Binary Vector:")
for vector in binary_vector:
    print(vector)

# print(f"LBP value at ({x}, {y}): {lbp_value}")
# print(lbp_result)

# Tính toán LBP descriptor cho mỗi điểm ảnh
lbp_descriptor_matrix = np.zeros_like(image_array)
for i in range(image_array.shape[0]):
    for j in range(image_array.shape[1]):
        lbp_descriptor_matrix[i, j] = get_lbp_pixel(image_array, i, j)
print("LBP Descriptor Matrix:")
print(lbp_descriptor_matrix)

# Tính histogram của các mẫu nhị phân
histogram = np.histogram(binary_vector, bins=256, range=(0, 256))[0]
print("Histogram: ")
print(histogram)