import os
import numpy as np
import torch
from pyexpat import features

from OpenShapeInference import OpenShapeInference

SRC_PATH = "OpenShape_code/src"

if __name__ == '__main__':
    openshape = OpenShapeInference(
        # model_repo="OpenShape/openshape-pointbert-vitg14-rgb",
        # clip_model='ViT-L/14',
        # clip_pretrained='openai',
        src_path=SRC_PATH,
        device="cpu",
    )

    # obj_path = "D:\\private\\objects\\00c19c25-bd0d-4355-8d84-e465f3f3eadd\\normalized_model.obj"
    # print("Test trich xuat dac trung mo hinh 3d")
    # shape_features = openshape.extract_shape_features(obj_path, 10000)
    # print(shape_features.shape, " Success!")

    features = np.load("G:\\My Drive\public_data_numpy_10000\\10daa874-0936-4bd6-9be2-3bbe831d1bcd\\4d698773-946c-4d46-8b2a-3aec6b9e74f3\\normalized_model.npy")
    features = torch.tensor(features).unsqueeze(0)

    # Swap trục Y và Z trong dữ liệu XYZ
    swap_features = features.clone()
    swap_features[:, :, [1, 2]] = swap_features[:, :, [2, 1]]

    xyz = swap_features[:, :, :3]  # [1, num_points, 3]
    shape_feat = openshape.model(xyz, swap_features)
    print(shape_feat.shape, shape_feat)

    texts = ["A white refridge", "Chair", "Table", "Bed"]

    for text in texts:
        text_embed = openshape.extract_text_features(text)
        print(f"{text}: {openshape.shape_to_text_similarity(shape_feat, text_embed)}")