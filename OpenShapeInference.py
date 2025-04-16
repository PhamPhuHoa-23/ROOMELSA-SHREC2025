import sys

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import re
import os
from PIL import Image
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import importlib.util
import open3d as o3d
from omegaconf import omegaconf, OmegaConf
from sympy.sets.sets import simplify_intersection


class OpenShapeInference:
    """
    Class for inference with OpenShape model (PointBERT variant) for windows
    """
    def __init__(self,
                 model_repo="OpenShape/openshape-pointbert-vitg14-rgb",
                 clip_model='ViT-bigG-14',
                 clip_pretrained='laion2b_s39b_b160k',
                 device=None,
                 src_path=None):
        """

        :param model_repo: HuggingFace repo path
        :param clip_model: Clip model name
        :param clip_pretrained: Clip pretrained model name
        :param device: Device to use
        :param src_path: Path to OpenShape src folder
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_repo = model_repo

        if src_path is not None:
            if src_path not in sys.path:
                sys.path.append(src_path)

        else:
            current_dir = os.getcwd()
            if os.path.exists(os.path.join(current_dir, 'src')):
                sys.path.append(os.path.join(current_dir, 'src'))
                if src_path not in sys.path:
                    sys.path.append(src_path)

        self.models_module = self._import_models(src_path)

        print("Dang tai mo hinhf OpenShape")
        self.config = self._load_config()
        self.model = self._load_model()
        self.model.eval()

        print("Dang tai mo hinh CLIP")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.clip_model.to(self.device).eval()

    def _import_models(self, src_path):
        import OpenShape_code.src.models as models
        return models

    def _load_config(self):
        config = OmegaConf.create(
            {
                'model': {
                    'name': 'PointBERT',
                    'in_channel': 6,
                    'out_channel': 1280,
                    'scaling': 4,
                    'use_dense': True
                }
            }
        )

        return config

    def _load_model(self):
        if hasattr(self.models_module, 'make'):
            model = self.models_module.make(self.config).to(self.device)

        else:
            raise NotImplementedError("Module models missing")

        checkpoint_path = hf_hub_download(repo_id=self.model_repo, filename="model.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k, v in checkpoint['state_dict'].items():
            if re.search('module', k):
                model_dict[re.sub(pattern, "", k)] = v
            else:
                model_dict[k] = v

        model.load_state_dict(model_dict)
        return model

    def _normalize_pc(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid

        max_norm = np.max(np.linalg.norm(pc, axis=1))
        if max_norm > 1e-6:
            pc =  pc / max_norm

        return pc

    def load_obj_as_pointcloud(self,
                               obj_path,
                               num_points=10000,
                               normalize=True,
                               y_up=True):
        mesh = o3d.io.read_triangle_mesh(obj_path)

        if len(mesh.vertices) == 0:
            raise ValueError('Khong tim that diem nao')

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        xyz = np.asarray(pcd.points).astype(np.float32)
        print(pcd, "hehe")
        rgb = np.asarray(pcd.colors).astype(np.float32)
        if y_up:
            # Swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]

        if normalize:
            xyz = self._normalize_pc(xyz)
            # rgb = self._normalize_pc(rgb)

        print(xyz.shape)
        print(rgb)
        features = np.concatenate((xyz, rgb), axis=1)
        # features = xyz
        xyz_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        return xyz_tensor, features_tensor

    @torch.no_grad()
    def extract_shape_features(self, obj_path, numpoints=1280):
        xyz, features = self.load_obj_as_pointcloud(obj_path)

        shape_features = self.model(torch.concatenate([xyz, features]))
        shape_features = F.normalize(shape_features, dim=-1)

        return shape_features

    @torch.no_grad()
    def extract_text_features(self, texts):
        text_tokens = open_clip.tokenize(texts).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=1)
        return text_features

    @torch.no_grad()
    def extract_image_features(self, image_paths):
        images = [Image.open(image_path) for image_path in image_paths]
        image_tensors = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
        image_features = self.clip_model.encode_image(image_tensors)
        image_features = F.normalize(image_features, dim=1)
        return image_features

    def shape_to_text_similarity(self, shape_features, text_features):
        return shape_features @ text_features.T

    def shape_to_image_similarity(self, shape_features, text_features):
        return shape_features @ text_features.T

    def text_to_shape_retrieval(self,
                                obj_paths,
                                text_query,
                                num_points=10000,
                                top_k=5):
        text_features = self.extract_text_features([text_query])

        results = []
        for path in obj_paths:
            shape_features = self.extract_shape_features(
                obj_path=path,
                num_points=num_points,
            )

            similarity = self.shape_to_text_similarity(shape_features, text_features)
            results.append([path, similarity])

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def image_to_shape_retrieval(self,
                                 obj_paths,
                                 image_path,
                                 num_points=1280,
                                 top_k=5):
        images_features = self.extract_image_features([image_path])

        results = []
        for path in obj_paths:
            shape_features = self.extract_shape_features(
                obj_path=path,
                num_points=num_points,
            )

            similarity = self.shape_to_image_similarity(shape_features, images_features)
            results.append([path, similarity])

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def shape_to_shape_similarity(self,
                                  obj_path1,
                                  obj_path2,
                                  num_points=1280,
                                  top_k=5):
        shape_features1 = self.extract_shape_features(obj_path1, numpoints=num_points)
        shape_features2 = self.extract_shape_features(obj_path2, numpoints=num_points)

        similarity = (shape_features1 @ shape_features2.T).cpu().item()

        return similarity


