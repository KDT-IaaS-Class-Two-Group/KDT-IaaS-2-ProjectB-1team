import cv2 as cv
img = cv.imread("./path/to/image.jpg")

# 이미지가 제대로 읽혔는지 확인
if img is None:
    print(f"Error: Could not open or find the image at {img}")
    
cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the windowx