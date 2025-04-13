import sys
import os
import cv2
import warnings

warnings.filterwarnings('ignore')

from PanoFormer.PanoFormer.PanoFormerDepthEstimator import PanoFormerDepthEstimator
# Điều chỉnh đường dẫn này nếu cần
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

if __name__ == '__main__':
    panoformer = PanoFormerDepthEstimator(weights_path="PanoFormer/PanoFormer/tmp/panodepth/models/weights_pretrain")
    # panoformer = PanoFormerDepthEstimator(weights_path="C:\\Users\\admin\OneDrive - VNU-HCMUS\\Documents")

    image_path = "G:\\My Drive\\public_data\\1baefb33-2a8f-49b7-9214-a3c06eb98559\\0_colors.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = panoformer.predict_depth(image)
    output_path = "G:\\My Drive\\public_data\\1baefb33-2a8f-49b7-9214-a3c06eb98559\\0_colors_depth.png"
    panoformer.save_depth_map(depth_map, output_path)
