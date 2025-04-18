import sys
import os
import cv2
import warnings
import torch
import numpy as np
import random
warnings.filterwarnings('ignore')

from PanoFormer.PanoFormer.PanoFormerDepthEstimator import PanoFormerDepthEstimator
# Điều chỉnh đường dẫn này nếu cần
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Đảm bảo các thao tác CUDA là xác định
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # panoformer = PanoFormerDepthEstimator(weights_path="PanoFormer/PanoFormer/tmp_s2d3d/panodepth/models/weights")
    panoformer = PanoFormerDepthEstimator(weights_path="PanoFormer/PanoFormer/tmp/panodepth/models/weights_pretrain")

    # panoformer = PanoFormerDepthEstimator(weights_path="C:\\Users\\admin\OneDrive - VNU-HCMUS\\Documents")

    fol = "21bce930-fd0c-4ef3-be2d-6cdd5bb3a8c7"
    image_path = f"G:\\My Drive\\public_data\\{fol}\\0_colors.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = panoformer.predict_depth(image)
    pcd = panoformer.to_point_cloud(
        depth_map=depth_map,
        rgb_image=image,
        filter_outlier=False,
    )
    panoformer.visualize_point_cloud(pcd)
