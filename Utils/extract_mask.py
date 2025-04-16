import cv2
import numpy as np
import argparse


def extract_mask_regions(image_path, color_lower, color_upper, output_path):
    """
    Trích xuất vùng mask dựa trên phạm vi màu và lưu thành file npy

    Tham số:
        image_path (str): Đường dẫn đến ảnh panorama
        color_lower (tuple): Giới hạn dưới của màu RGB (R, G, B)
        color_upper (tuple): Giới hạn trên của màu RGB (R, G, B)
        output_path (str): Đường dẫn để lưu file npy
    """
    # Đọc ảnh
    img = cv2.imread(image_path)

    if img is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return

    # Chuyển từ BGR (OpenCV) sang RGB để so sánh
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Tạo mask dựa trên phạm vi màu
    mask = cv2.inRange(img_rgb, color_lower, color_upper)

    # Tìm các contour trong mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo mask rỗng để vẽ các contour
    result_mask = np.zeros_like(mask)

    # Tạo dictionary để lưu các vùng mask và tọa độ bounding box của chúng
    mask_regions = {}

    for i, contour in enumerate(contours):
        # Lọc ra các contour nhỏ (nhiễu)
        if cv2.contourArea(contour) < 100:
            continue

        # Vẽ contour lên mask kết quả
        cv2.drawContours(result_mask, [contour], -1, 255, -1)

        # Tìm bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Lưu thông tin cho vùng này
        mask_regions[f'region_{i}'] = {
            'bbox': (x, y, w, h),
            'mask': result_mask[y:y + h, x:x + w].copy(),
            'contour': contour
        }

    # Lưu mask_regions dưới dạng file npy
    np.save(output_path, mask_regions)

    print(f"Đã lưu {len(mask_regions)} vùng mask vào {output_path}")

    # Vẽ và hiển thị kết quả để kiểm tra (có thể bỏ nếu không cần)
    debug_img = img.copy()
    for region_name, region_data in mask_regions.items():
        x, y, w, h = region_data['bbox']
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, region_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Lưu ảnh debug để kiểm tra
    cv2.imwrite('debug_mask_regions.jpg', debug_img)

    return mask_regions


def main():
    parser = argparse.ArgumentParser(description='Trích xuất vùng mask từ ảnh panorama')
    parser.add_argument('--image', required=True, help='Đường dẫn đến ảnh panorama')
    parser.add_argument('--output', default='mask_regions.npy', help='Đường dẫn để lưu file npy')
    parser.add_argument('--r_min', type=int, default=135, help='Giá trị tối thiểu của kênh R')
    parser.add_argument('--g_min', type=int, default=206, help='Giá trị tối thiểu của kênh G')
    parser.add_argument('--b_min', type=int, default=235, help='Giá trị tối thiểu của kênh B')
    parser.add_argument('--r_max', type=int, default=135, help='Giá trị tối đa của kênh R')
    parser.add_argument('--g_max', type=int, default=206, help='Giá trị tối đa của kênh G')
    parser.add_argument('--b_max', type=int, default=235, help='Giá trị tối đa của kênh B')

    args = parser.parse_args()

    # Màu xanh lam (giá trị mặc định được điều chỉnh cho màu xanh dương trong ảnh của bạn)
    color_lower = (args.r_min, args.g_min, args.b_min)
    color_upper = (args.r_max, args.g_max, args.b_max)

    extract_mask_regions(args.image, color_lower, color_upper, args.output)


if __name__ == "__main__":
    main()