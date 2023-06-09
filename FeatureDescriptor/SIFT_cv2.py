import cv2


#Lấy ảnh từ máy
img_path = r"C:\Users\nomdd\OneDrive\Pictures\Screenshots\quat.jpg"
image = cv2.imread(img_path)

gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(image, None)

image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

cv2.imshow('anh goc', image)
cv2.imshow('anh da ve keypoint', image_with_keypoints)
cv2.resizeWindow('anh goc', 800, 700)
cv2.resizeWindow('anh da ve keypoint', 800, 700)
cv2.moveWindow('anh goc', 0, 0)
cv2.moveWindow('anh da ve keypoint', 800, 0)
cv2.waitKey(0)
cv2.destroyAllWindows()



