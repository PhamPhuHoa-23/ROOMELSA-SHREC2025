import cv2
import numpy as np

# Biến toàn cục để lưu giá trị màu
clicked_colors = []


def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lấy màu tại vị trí click
        b, g, r = img[y, x]
        color_rgb = (r, g, b)
        clicked_colors.append(color_rgb)
        print(f"RGB at position ({x},{y}): {color_rgb}")

        # Hiển thị màu đã chọn
        color_display = np.zeros((100, 200, 3), np.uint8)
        color_display[:] = (b, g, r)
        cv2.imshow('Selected Color', color_display)


# Đọc ảnh
image_path = r'D:\private\scenes\08df38e7-b9ec-40d1-8652-b1857959a6c7\masked.png'  # Thay đổi đường dẫn tới ảnh của bạn
img = cv2.imread(image_path)

if img is None:
    print(f"Không thể đọc ảnh từ {image_path}")
    exit()

# Thiết lập cửa sổ và callback
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', pick_color)

# Hiển thị ảnh và đợi người dùng click
print("Click vào ảnh để lấy mẫu màu. Nhấn 'q' để thoát.")
while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hiển thị tất cả màu đã chọn
if clicked_colors:
    print("\nTất cả màu đã chọn:")
    for i, color in enumerate(clicked_colors):
        print(f"{i + 1}. RGB: {color}")

cv2.destroyAllWindows()

# [135, 206, 235]