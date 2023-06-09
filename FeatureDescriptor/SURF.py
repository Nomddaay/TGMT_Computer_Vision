import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def surf(image_array, num_octaves=4, num_scales=5, threshold=0.1):
    # Chuyển đổi ảnh sang ảnh xám
    gray_image = image_array.astype(np.float32)

    # Tạo Gaussian Pyramid
    pyramid = create_gaussian_pyramid(gray_image, num_octaves, num_scales)

    # Tính toán Đạo hàm Gauss
    dx, dy = compute_gradients(pyramid)

    # Phát hiện điểm đặc trưng
    keypoints = detect_keypoints(dx, dy, threshold)

    return keypoints

def create_gaussian_pyramid(image, num_octaves, num_scales):
    pyramid = []
    for octave in range(num_octaves):
        octave_scale = []
        for scale in range(num_scales):
            sigma = 1.6 * 2 ** (scale / num_scales)
            smoothed = gaussian_filter(image, sigma)
            octave_scale.append(smoothed)
        pyramid.append(octave_scale)
        image = image[::2, ::2]  # Lấy mẫu ảnh để tạo octave mới
    return pyramid

def compute_gradients(pyramid):
    dx_pyramid = []
    dy_pyramid = []
    for octave_scale in pyramid:
        octave_dx = []
        octave_dy = []
        for scale_image in octave_scale:
            dx, dy = np.gradient(scale_image)
            octave_dx.append(dx)
            octave_dy.append(dy)
        dx_pyramid.append(octave_dx)
        dy_pyramid.append(octave_dy)
    return dx_pyramid, dy_pyramid

def detect_keypoints(dx_pyramid, dy_pyramid, threshold):
    keypoints = []
    for octave in range(len(dx_pyramid)):
        for scale in range(len(dx_pyramid[octave])):
            dx = dx_pyramid[octave][scale]
            dy = dy_pyramid[octave][scale]
            for i in range(1, dx.shape[0]-1):
                for j in range(1, dx.shape[1]-1):
                    if is_extremum(dx, dy, octave, scale, i, j) and is_corner(dx, dy, octave, scale, i, j, threshold):
                        keypoints.append((octave, scale, i, j))
    return keypoints

def is_extremum(dx, dy, octave, scale, i, j):
    value = dx[i, j]
    for o in range(max(0, octave-1), min(octave+2, len(dx))):
        for s in range(max(0, scale-1), min(scale+2, len(dx[o]))):
            if dx[o, s, i, j] > value or dx[o, s, i, j] < value:
                return False
    return True

def is_corner(dx, dy, octave, scale, i, j, threshold):
    Dxx = dx[octave, scale, i, j+1] - 2*dx[octave, scale, i, j] + dx[octave, scale, i, j-1]
    Dyy = dy[octave, scale, i+1, j] - 2*dy[octave, scale, i, j] + dy[octave, scale, i-1, j]
    Dxy = (dx[octave, scale, i+1, j+1] - dx[octave, scale, i+1, j-1] - dx[octave, scale, i-1, j+1] + dx[octave, scale, i-1, j-1]) / 4
    tr = Dxx + Dyy
    det = Dxx * Dyy - Dxy**2
    curvature_ratio = (tr**2) / det if det != 0 else float('inf')
    if det < 0 or curvature_ratio > threshold:
        return False
    return True

# Đường dẫn tới ảnh trên máy
image_path = r'C:\Users\nomdd\OneDrive\Pictures\Screenshots\cay.jpg'

# Đọc ảnh và chuyển đổi thành mảng numpy
image = Image.open(image_path)
image_array = np.array(image)

# Thực hiện SURF trên ảnh
keypoints = surf(image_array)

# In ra số lượng điểm đặc trưng được tìm thấy
print(f"Số lượng điểm đặc trưng: {len(keypoints)}")
