import logging
import os
import json
import torch
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from flask import views
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)


class Model3DDataset(Dataset):
    """
    Dataset class for the 3D model
    """
    def __init__(self, json_path,
                 root_dir=None,
                 transform=None,
                 device=None,
                 load_images=True,
                 load_models=True):
        """

        :param json_path: Path to the json file
        :param root_dir: Path to the root directory
        :param transform: Transformation applied to the images
        :param device: Device to use
        :param load_images: Is the image loading
        :param load_models: Is the model loading
        """
        self.json_path = os.path.abspath(json_path)

        if root_dir is None:
            self.root_dir = os.path.dirname(self.json_path)
        else:
            self.root_dir = os.path.abspath(root_dir)

        with open(json_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)

        self.models = []

        for dir_level1, level1_data in self.data.items():
            for dir_level2, model_data in level1_data.items():
                model_data['dir_level1'] = dir_level1
                model_data['dir_level2'] = dir_level2
                self.models.append(model_data)

        self.transform = transform
        self.device = device
        self.load_images = load_images
        self.load_models = load_models

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        model_data = self.models[idx]
        result = {
            'dir_level1': model_data['dir_level1'],
            'dir_level2': model_data['dir_level2']
        }

        if self.load_images and 'image_path' in model_data:
            image_path = os.path.join(self.root_dir, model_data['image_path'])
            print(f"Image path: {image_path}")

            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                result['image'] = image

            except Exception as e:
                print(f"Error while loading image: {image_path}")
                result['image'] = None

        if self.load_models and 'obj_path' in model_data:
            obj_path = os.path.join(self.root_dir, model_data['obj_path'])
            print(f"Object path: {obj_path}")

            try:
                verts, faces, aux = load_obj(
                    obj_path,
                    device=self.device,
                    load_textures=True,
                    create_texture_atlas=True
                )

                center = verts.mean(0)
                verts = verts - center
                scale = verts.abs().max()
                verts = verts / scale

                if aux.texture_images and 'texture_path' in model_data:
                    textures = aux.texture_atlats
                    mesh = Meshes(
                        verts=[verts],
                        faces=faces,
                        textures=textures
                    )

                result['mesh'] = mesh
                result['verts'] = verts
                result['faces'] = faces

            except Exception as e:
                print(f"Error while loading image: {obj_path}")
                result['mesh'] = None
                result['verts'] = None
                result['faces'] = None

        for key, value in model_data.items():
            if key not in result and key not in ['dir_level1', 'dir_level2']:
                result[key] = value

        return result

class DataCreator:
    def __init__(self,
                 root_dir=None,
                 output_dir=None,
                 image_size=256,
                 num_views=12,
                 device=None,
                 view_config="openshape",
                 dist=2.0,
                 logger=None,
                 json_path=None):
        """
        Initilize the DataCreator

        :param root_dir: Path to the root directory
        :param output_dir: Path to the output directory
        :param image_size: Image size
        :param num_views: Number of views
        :param device: Device to use
        :param view_config: "openshape", "uniform", "custom"
        :param dist: Distance between camera and model
        :param looger: Logger
        :param json_path: Path to the json file
        """

        self.logger = logger if logger else self._setup_logger()

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.root_dir = root_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.num_views = num_views
        self.dist = dist
        self.json_path = json_path

        self.view_config = view_config
        self.views = self._setup_view_config(view_config)

        self.renderer = None

        self.dataset_json = None
        if json_path and os.path.exists(json_path):
            self._load_json_dataset()

    def _setup_logger(self):
        logger = logging.getLogger("DataCreator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

        return logger

    def _setup_view_config(self, view_config):
        if view_config == "openshape":
            return [
                (0, 0),
                (90, 0),
                (180, 0),
                (270, 0),

                (0, 30),
                (90, 30),
                (180, 30),
                (270, 30),

                (0, -30),
                (90, -30),
                (180, -30),
                (270, -30)
            ]
        return []

    def _load_json_dataset(self):
        with open(self.json_path, "r", encoding='utf-8') as f:
            self.dataset_json = json.load(f)
            self.logger.info(f"Loaded dataset from {self.json_path}")

    def _setup_renderer(self, image_size=None):
        """
        Setup the renderer

        :param image_size:
        :return: MeshRenderer
        """
        if image_size is None:
            image_size = self.image_size

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device)
        )

        return renderer

    def load_model(self, obj_path):
        """
        Load the model

        :param obj_path: Path to the obj file
        :return: Mesh with Texture
        """
        dir_path = os.path.dirname(obj_path)

        verts, faces, aux = load_obj(
            obj_path,
            device=self.device,
            load_textures=True,
            create_texture_atlas=True
        )

        texture_image = list(aux.texture_images.values())[0]
        if texture_image is None:
            texture_path = os.path.join(dir_path, "texture.png")
            texture_image = Image.open(texture_path).convert('RGB')
            texture_image = torch.from_numpy(np.array(texture_image) / 255.0)[..., :3].to(self.device)

        if (texture_image is not None and aux.verts_uvs is not None):
                # and aux.faces_uvs is not None):
            verts_uvs = aux.verts_uvs.to(self.device)
            # faces_uvs = aux.faces_uvs.to(self.device)
            verts_rgb = torch.ones_like(verts)[None]
            # textures = TexturesUV(
            #     maps=[texture_image],
            #     faces_uvs,
            #     verts_uvs=verts_uvs
            # )
            textures = TexturesVertex(verts_features=verts_rgb)

        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=textures
        )

        mesh = self.normalize_mesh(mesh)

        return mesh

    def normalize_mesh(self, mesh):
        """
        Normalize the mesh
        :param mesh: Mesh
        :return: Normalized mesh
        """
        verts = mesh.verts_packed()
        center = verts.mean(0)
        verts = verts - center
        scale = verts.abs().max()
        verts = verts / scale

        new_verts_list = []
        for v in mesh.verts_list():
            new_v = v - center
            new_v = new_v / scale
            new_verts_list.append(new_v)

        normalized_mesh = mesh.update_padded(np.asarray(new_verts_list))

        return normalized_mesh

    def render_view(self, mesh, azim, elev, dist=None, image_size=None):
        """
        Render the view

        :param mesh: Mesh
        :param azim: Theta
        :param elev: Phi
        :param dist: Distance between camera and model
        :param image_size: Image size
        :return: Rendered view
        """
        if dist is None:
            dist = self.dist

        if image_size is None:
            image_size = self.image_size

        if self.renderer is None or self.renderer.raserizer.rasterizer_settings.image_size != image_size:
            self.renderer = self._setup_renderer(image_size)

        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 3.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.7, 0.7, 0.7]],
            specular_color=[[0.3, 0.3, 0.3]]
        )

        self.renderer.shader.cameras = cameras
        self.renderer.shader.lights = lights
        self.renderer.rasterizer.rasterizer.cameras = cameras

        images = self.renderer.render(mesh)
        return images[..., :3]

    def capture_views(self, mesh, output_dir=None, views=None, image_size=None):
        """
        Capture views

        :param mesh: Mesh
        :param output_dir: Path to the output directory
        :param views: List of views
        :param image_size: Image size
        :return: List of path to images
        """
        if output_dir is None:
            output_dir = self.output_dir

        if views is None:
            views = self.views

        if image_size is None:
            image_size = self.image_size

        os.makedirs(output_dir, exist_ok=True)

        image_paths = []

        for i, (azim, elev) in enumerate(views):
            image = self.render_view(mesh, azim, elev, image_size)
            image_np = image.detach().cpu().numpy()
            image_paths = (image_np * 255).astype(np.uint8)

            image_pil = Image.fromarray(image_paths)
            image_path = os.path.join(output_dir, f"view_{i:02d}_azim_{azim}_elev_{elev}.png")
            image_pil.save(image_path)

            self.logger.info(f"Saved view {i} at {image_path}")

        return image_paths

    def process_model(self, obj_path, output_dir=None):
        """
        Process model

        :param obj_path: Path to the model
        :param output_dir: Path to the output directory
        :return: List of paths to images
        """

        if output_dir is None:
            if self.output_dir is None:
                raise ValueError("output_dir must be set")

            model_name = os.path.splitext(os.path.basename(obj_path))[0]
            output_dir = os.path.join(self.output_dir, model_name)

        mesh = self.load_model(obj_path)
        image_paths = self.capture_views(mesh, output_dir=output_dir)
        self.logger.info(f"Succesfully processed model: {obj_path}")

        return image_paths

    def find_obj_files(self, directory=None):
        """"""
        if directory is None:
            if self.root_dir is None:
                raise ValueError("root_dir must be set")

            directory = self.root_dir

        obj_files = []
        for root, dirs,files in os.walk(directory):
            for file in files:
                if file.endswith(".obj"):
                    obj_files.append(os.path.join(root, file))

        self.logger.info(f"Found {len(obj_files)} obj files")
        return obj_files

    def process_directory(self, directory=None, output_base_dir=None, recursive=True):
        """

        :param directory: Thư mục cần xử lý
        :param output_base_dir: Thư mục đầu ra gốc
        :param recursive: Có tìm kiếm trong các thư mục con không
        :return: Dict với key là đường dẫn đến file obj và value là danh sách các ảnh đã được capture
        """
        if directory is None:
            if self.root_dir is None:
                raise ValueError("root_dir must be set")
            directory = self.root_dir

        if output_base_dir is None:
            if self.output_dir is None:
                raise ValueError("output_dir must be set")
            output_base_dir = self.output_dir

        if recursive:
            obj_files = self.find_obj_files(directory=directory)
        else:
            obj_files = glob.glob(os.path.join(directory, "*.obj"))
            self.logger.info(f"Found {len(obj_files)} obj files (non recursive)")

        results = {}

        for i, obj_path in enumerate(obj_files, desc="Processing models"):
            model_name = os.path.splitext(os.path.basename(obj_path))[0]
            model_dir = os.path.dirname(obj_path)
            relative_path = os.path.relpath(model_dir, directory)
            output_dir = os.path.join(output_base_dir, relative_path, model_name)

            image_paths = self.process_model(obj_path, output_dir)
            results[obj_path] = image_paths

        return results

    def process_json_dataset(self, output_base_dir=None, root_dir=None):
        """
        Xử lý tất cả mô hinhf từ file json dataset

        :param output_base_dir: Thư mục đầu ra gốc
        :param root_dir: Đường dẫn gốc của dataset
        :return: Dict, kết quả xử lý
        """

        if self.dataset_json is None:
            if self.json_path is None:
                raise ValueError("json_path must be set")
            self._load_json_dataset()

        if output_base_dir is None:
            if self.output_dir is None:
                raise ValueError("output_dir must be set")
            output_base_dir = self.output_dir

        if root_dir is None:
            if self.root_dir is None:
                raise ValueError("root_dir must be set")
            root_dir = self.root_dir

        os.makedirs(output_base_dir, exist_ok=True)

        models_to_process = []
        for dir_level1, level1_data in self.dataset_json.items():
            for dir_level2, model_data in level1_data.items():
                if 'obj_path' in model_data:
                    obj_path = os.path.join(root_dir, model_data['obj_path'])
                    rel_output_dir = os.path.join(dir_level1, dir_level2)
                    output_dir = os.path.join(output_base_dir, rel_output_dir)
                    models_to_process.append({
                        'obj_path': obj_path,
                        'output_dir': output_dir,
                        'dir_level1': dir_level1,
                        'dir_level2': dir_level2,
                    })

        self.logger.info(f"Found {len(models_to_process)} models to process")

        results = {}
        for i, model_info in enumerate(tqdm(models_to_process, desc="Processing models")):
            obj_path = model_info['obj_path']
            output_dir =  model_info['output_dir']
            dir_level1 = model_info['dir_level1']
            dir_level2 = model_info['dir_level2']

            if dir_level1 not in results:
                results[dir_level1] = {}

            os.makedirs(output_dir, exist_ok=True)
            image_paths = self.process_model(obj_path, output_dir)
            view_paths = []
            for img_path in image_paths:
                rel_path = os.path.relpath(img_path, output_base_dir)
                view_paths.append(img_path)

            results[dir_level1][dir_level2] = {
                'views': view_paths,
                'success': True
            }

        results_file = os.path.join(output_base_dir, "processing_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        self.logger.info(f"Saving processing {len(models_to_process)} models")
        total_models = sum(len(dirs) for dirs in results.values())
        successful_models = sum(sum(1 for model in dirs.values() if model.get('success', False))
                                for dirs in results.values())

        self.logger.info(f"Processed {total_models} models, {successful_models} successful, "
                         f"{total_models - successful_models} failed")

        return results

    def batch_process(self, obj_files, output_dirs=None, batch_size=10):
        """
        Xử lý một loạt các obj cho trước

        :param obj_files: Danh sách các đường dẫn đến file obj
        :param output_dirs: Danh sách các thư mục đầu ra
        :param batch_size: Số luognjw mô hình xử lý môi lần
        :return: Dictionary với key là đường dẫn obj và value là danh sách các ảnh đã tạo
        """
        if output_dirs is None:
            output_dirs = [None] * len(obj_files)

        results = {}

        for i in range(0, len(obj_files), batch_size):
            batch_files = obj_files[i:i + batch_size]
            batch_outputs = output_dirs[i:i + batch_size]

            for j, (obj_path, output_dir) in enumerate(zip(batch_files, batch_outputs)):
                self.logger.info(
                    f"Processing batch {i // batch_size + 1}, model {j + 1}/{len(batch_files)}: {obj_path}")
                image_paths = self.process_model(obj_path, output_dir)
                results[obj_path] = image_paths

        return results

    def save_dataset_metadata(self, output_path, metadata):
        """
        Lưu metadata của dataset

        Args:
            output_path (str): Đường dẫn để lưu file metadata
            metadata (dict): Metadata cần lưu
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved dataset metadata to: {output_path}")

    def export_view_configs(self, output_path):
        """
        Xuất cấu hình góc nhìn để sử dụng lại sau này

        Args:
            output_path (str): Đường dẫn để lưu file cấu hình
        """
        config = {
            'view_config': self.view_config,
            'views': self.views,
            'num_views': self.num_views,
            'image_size': self.image_size,
            'dist': self.dist
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Exported view configs to: {output_path}")

    def import_view_configs(self, config_path):
        """
        Nhập cấu hình góc nhìn từ file

        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'views' in config:
                self.views = config['views']
                self.num_views = len(self.views)

            if 'view_config' in config:
                self.view_config = config['view_config']

            if 'image_size' in config:
                self.image_size = config['image_size']

            if 'dist' in config:
                self.dist = config['dist']

            self.logger.info(f"Imported view configs from: {config_path}")
            self.logger.info(f"Current settings: view_config={self.view_config}, num_views={self.num_views}, "
                             f"image_size={self.image_size}, dist={self.dist}")

            # Reset renderer to apply new settings
            self.renderer = None

        except Exception as e:
            self.logger.error(f"Error importing view configs: {str(e)}")

    def create_dataset_json(self, root_dir, output_json, find_description=False):
        """
        Tạo file JSON chứa thông tin về tất cả các mô hình trong dataset

        Args:
            root_dir (str): Đường dẫn đến thư mục gốc của dataset
            output_json (str): Đường dẫn để lưu file JSON
            find_description (bool): Tìm file mô tả nếu có (mặc định: False)

        Returns:
            dict: Dữ liệu JSON đã tạo
        """
        # Chuẩn hóa đường dẫn gốc
        root_dir = os.path.abspath(root_dir)

        # Tạo cấu trúc dữ liệu
        dataset_json = {}

        # Đếm số lượng mô hình để hiển thị tiến trình
        total_models = 0
        for dir_level1 in os.listdir(root_dir):
            level1_path = os.path.join(root_dir, dir_level1)
            if os.path.isdir(level1_path):
                for dir_level2 in os.listdir(level1_path):
                    level2_path = os.path.join(level1_path, dir_level2)
                    if os.path.isdir(level2_path):
                        if any(file.endswith('.obj') for file in os.listdir(level2_path)):
                            total_models += 1

        self.logger.info(f"Found {total_models} model directories")

        # Duyệt qua cấu trúc thư mục
        with tqdm(total=total_models, desc="Creating dataset JSON") as pbar:
            for dir_level1 in sorted(os.listdir(root_dir)):
                level1_path = os.path.join(root_dir, dir_level1)

                # Chỉ xử lý thư mục
                if not os.path.isdir(level1_path):
                    continue

                # Tạo entry cho thư mục cấp 1
                dataset_json[dir_level1] = {}

                for dir_level2 in sorted(os.listdir(level1_path)):
                    level2_path = os.path.join(level1_path, dir_level2)

                    # Chỉ xử lý thư mục
                    if not os.path.isdir(level2_path):
                        continue

                    # Kiểm tra xem có file .obj nào không
                    obj_files = [f for f in os.listdir(level2_path) if f.endswith('.obj')]
                    if not obj_files:
                        continue

                    # Tạo entry cho thư mục cấp 2
                    object_data = {}

                    # Tìm các file theo thứ tự ưu tiên
                    # 1. normalized_model.obj (nếu có)
                    # 2. model.obj (nếu có)
                    # 3. Bất kỳ file .obj nào khác
                    obj_path = None
                    if "normalized_model.obj" in obj_files:
                        obj_path = os.path.join(level2_path, "normalized_model.obj")
                    elif "model.obj" in obj_files:
                        obj_path = os.path.join(level2_path, "model.obj")
                    else:
                        obj_path = os.path.join(level2_path, obj_files[0])

                    # Chuyển đổi đường dẫn tuyệt đối thành đường dẫn tương đối so với root_dir
                    rel_obj_path = os.path.relpath(obj_path, root_dir)
                    object_data["obj_path"] = rel_obj_path

                    # Tìm file texture.png
                    texture_path = os.path.join(level2_path, "texture.png")
                    if os.path.exists(texture_path):
                        rel_texture_path = os.path.relpath(texture_path, root_dir)
                        object_data["texture_path"] = rel_texture_path

                    # Tìm file image.png hoặc bất kỳ file ảnh nào khác
                    image_files = [f for f in os.listdir(level2_path) if
                                   f.endswith(('.png', '.jpg', '.jpeg')) and f != "texture.png"]
                    if image_files:
                        image_path = os.path.join(level2_path, image_files[0])
                        rel_image_path = os.path.relpath(image_path, root_dir)
                        object_data["image_path"] = rel_image_path

                    # Tìm file mô tả nếu được yêu cầu
                    if find_description:
                        description_files = [f for f in os.listdir(level2_path) if
                                             f.endswith(('.txt', '.md')) and "description" in f.lower()]
                        if description_files:
                            desc_path = os.path.join(level2_path, description_files[0])
                            try:
                                with open(desc_path, 'r', encoding='utf-8') as f:
                                    description = f.read().strip()
                                object_data["description"] = description
                            except Exception as e:
                                self.logger.error(f"Error reading description file {desc_path}: {e}")
                                object_data["description"] = ""
                        else:
                            object_data["description"] = ""

                    # Thêm thông tin mô hình vào dataset
                    dataset_json[dir_level1][dir_level2] = object_data

                    # Cập nhật thanh tiến trình
                    pbar.update(1)

        # Lưu dữ liệu vào file JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved dataset JSON to: {output_json}")

        # Thống kê
        total_level1 = len(dataset_json)
        total_models = sum(len(dirs) for dirs in dataset_json.values())
        self.logger.info(f"Dataset statistics: {total_level1} level-1 directories, {total_models} models")

        # Cập nhật dataset_json
        self.dataset_json = dataset_json
        self.json_path = output_json

        return dataset_json

    def update_dataset_json(self, json_path=None, views_output_dir=None):
        """
        Cập nhật file JSON để thêm đường dẫn đến các góc nhìn đã tạo

        Args:
            json_path (str): Đường dẫn đến file JSON cần cập nhật, nếu None sẽ dùng self.json_path
            views_output_dir (str): Thư mục chứa các góc nhìn, nếu None sẽ dùng self.output_dir

        Returns:
            dict: Dữ liệu JSON đã cập nhật
        """
        if json_path is None:
            if self.json_path is None:
                raise ValueError("json_path must be specified either in the constructor or method call")
            json_path = self.json_path

        if views_output_dir is None:
            if self.output_dir is None:
                raise ValueError("views_output_dir must be specified either in the constructor or method call")
            views_output_dir = self.output_dir

        # Load JSON hiện tại
        if self.dataset_json is None or json_path != self.json_path:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    dataset_json = json.load(f)
                self.dataset_json = dataset_json
                self.json_path = json_path
            except Exception as e:
                self.logger.error(f"Error loading JSON file {json_path}: {str(e)}")
                return None
        else:
            dataset_json = self.dataset_json

        # Cập nhật thông tin góc nhìn
        views_updated = 0

        for dir_level1, level1_data in dataset_json.items():
            for dir_level2, model_data in level1_data.items():
                # Tạo đường dẫn đến thư mục chứa các góc nhìn
                views_dir = os.path.join(views_output_dir, dir_level1, dir_level2)

                if os.path.exists(views_dir):
                    # Tìm tất cả các file ảnh trong thư mục
                    view_files = []
                    for root, _, files in os.walk(views_dir):
                        for file in files:
                            if file.endswith(('.png', '.jpg', '.jpeg')):
                                file_path = os.path.join(root, file)
                                rel_path = os.path.relpath(file_path, views_output_dir)
                                view_files.append(rel_path)

                    if view_files:
                        model_data['views'] = view_files
                        views_updated += 1

        # Lưu JSON đã cập nhật
        updated_json_path = os.path.splitext(json_path)[0] + "_with_views.json"
        with open(updated_json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Updated {views_updated} models with view paths")
        self.logger.info(f"Saved updated JSON to: {updated_json_path}")

        return dataset_json

    def visualize_model(self, obj_path, save_path=None, show=True):
        """
        Hiển thị một mô hình 3D từ các góc nhìn khác nhau và lưu thành ảnh

        Args:
            obj_path (str): Đường dẫn đến file .obj
            save_path (str): Đường dẫn để lưu ảnh tổng hợp, nếu None sẽ không lưu
            show (bool): Có hiển thị ảnh không

        Returns:
            fig: Đối tượng Figure matplotlib
        """
        try:
            # Load mô hình
            mesh = self.load_model(obj_path)

            # Tạo hình ảnh từ các góc nhìn
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            axes = axes.flatten()

            for i, (azim, elev) in enumerate(self.views):
                # Render góc nhìn
                image = self.render_view(mesh, azim, elev)

                # Chuyển sang numpy để hiển thị
                image_np = image.detach().cpu().numpy()

                # Hiển thị ảnh
                axes[i].imshow(image_np)
                axes[i].set_title(f"Azim: {azim}, Elev: {elev}")
                axes[i].axis('off')

            plt.tight_layout()

            # Lưu ảnh nếu được yêu cầu
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved visualization to: {save_path}")

            # Hiển thị nếu được yêu cầu
            if show:
                plt.show()

            return fig

        except Exception as e:
            self.logger.error(f"Error visualizing model {obj_path}: {str(e)}")
            raise e

def get_dataloader(
        json_path,
        root_dir=None,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        transform=None,
        device=None,
        load_images=True,
        load_models=True
):
    """
    DataLoader for Model3DDataset

    :param json_path: Path to the json file
    :param root_dir: Path to the root directory
    :param batch_size: Batch size
    :param shuffle: Shuffle
    :param num_workers: Number of workers
    :param transform: Transformation applied to the images
    :param device: Device to use
    :param load_images: Is the image loading
    :param load_models: Is the model loading
    :return: DataLoader
    """

    dataset = Model3DDataset(
        json_path=json_path,
        root_dir=root_dir,
        transform=transform,
        device=device,
        load_images=load_images,
        load_models=load_models
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def collate_fn(batch):
    """
    Custom collate function

    :param batch: Batch of dataset
    :return: Dict
    """
    result = {}
    keys = batch[0].keys()

    for key in keys:
        if key == 'mesh':
            meshes = [
                item[key] for item in batch if item[key] is not None
            ]
            if meshes:
                result[key] = meshes[0].extend(meshes[1:] if len(meshes) > 1 else meshes[0])
            else:
                result[key] = None

        elif key == 'image' and batch[0][key] is not None:
            images = [item[key] for item in batch if item[key] is not None]

            if images and torch.is_tensor(images[0]):
                result[key] = torch.stack(images)
            else:
                result[key] = images

        else:
            result[key] = [item[key] for item in batch]

    return result



# if __name__ == '__main__':
#     import torch
#     import torchvision.transforms as transforms
#     from tqdm import tqdm
#
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     dataset = Model3DDataset(
#         json_path='G:\\My Drive\\public_data\\object.json',
#         transform=transform,
#         load_images=True,
#         load_models=False
#     )
#
#     print(f"Number of models: {len(dataset)}")
#
#     dataloader = get_dataloader(
#         json_path='G:\\My Drive\\public_data\\object.json',
#         batch_size=4,
#         transform=transform,
#         load_images=True,
#         load_models=False
#     )
#
#     # Kiểm tra một số batch
#     for i, batch in enumerate(tqdm(dataloader, desc="Kiểm tra DataLoader")):
#         if i >= 1:  # Chỉ kiểm tra 5 batch đầu tiên
#             break
#
#         print(f"Batch {i}:")
#         print(f"Kích thước batch: {len(batch['dir_level1'])}")
#         if 'image' in batch and batch['image'] is not None:
#             print(f"Kích thước ảnh trong batch: {batch['image'].shape}")

# Ví dụ sử dụng:
if __name__ == "__main__":
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(description='Tạo dữ liệu góc nhìn từ mô hình 3D')
        parser.add_argument('--root_dir', type=str, default=None,
                            help='Thư mục gốc chứa các mô hình 3D')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Thư mục đầu ra để lưu các ảnh render')
        parser.add_argument('--json_path', type=str, default=None,
                            help='Đường dẫn đến file JSON mô tả dataset (nếu có)')
        parser.add_argument('--image_size', type=int, default=256,
                            help='Kích thước ảnh đầu ra (mặc định: 256)')
        parser.add_argument('--view_config', type=str, default='openshape',
                            choices=['openshape', 'uniform', 'custom'],
                            help='Cấu hình góc nhìn (mặc định: openshape)')
        parser.add_argument('--create_json', action='store_true',
                            help='Tạo file JSON mô tả dataset')
        parser.add_argument('--process_json', action='store_true',
                            help='Xử lý dataset từ file JSON')
        parser.add_argument('--process_directory', action='store_true',
                            help='Xử lý tất cả các mô hình trong thư mục')
        parser.add_argument('--single_model', type=str, default=None,
                            help='Xử lý một mô hình duy nhất')
        parser.add_argument('--visualize', action='store_true',
                            help='Hiển thị mô hình từ các góc nhìn khác nhau')
        parser.add_argument('--cpu', action='store_true',
                            help='Sử dụng CPU thay vì GPU')
        return parser.parse_args()


    # Hàm main
    def main():
        # args = parse_args()


        # Thiết lập thiết bị
        # if args.cpu:
        #     device = torch.device("cpu")
        # else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Khởi tạo DataCreator
        creator = DataCreator(
            root_dir='G:\\My Drive\\public_data',
            output_dir='G:\\My Drive\\output',
            image_size=(224, 224),
            view_config='openshape',
            device=device,
            json_path='G:\\My Drive\\public_data\\objects.json',
        )

        # Tạo file JSON mô tả dataset
        # if args.create_json:
        #     if not args.root_dir:
        #         print("Error: --root_dir is required for --create_json")
        #         return

        output_json = 'G:\\My Drive\\public_data\\objects.json' if 'G:\\My Drive\\public_data\\objects.json' else os.path.join('G:\\My Drive\\output', "dataset.json")
        creator.create_dataset_json('G:\\My Drive\\public_data', output_json)

        # Xử lý dataset từ file JSON
        # elif args.process_json:
        #     if not args.json_path:
        #         print("Error: --json_path is required for --process_json")
        #         return
        #
        #     if not args.output_dir:
        #         print("Error: --output_dir is required for --process_json")
        #         return

        creator.process_json_dataset('G:\\My Drive\\output', 'G:\\My Drive\\public_data')

        # Xử lý tất cả các mô hình trong thư mục
        # elif args.process_directory:
        #     if not args.root_dir:
        #         print("Error: --root_dir is required for --process_directory")
        #         return
        #
        #     if not args.output_dir:
        #         print("Error: --output_dir is required for --process_directory")
        #         return

        creator.process_directory('G:\\My Drive\\public_data', 'G:\\My Drive\\output')

        # Xử lý một mô hình duy nhất
        # elif args.single_model:
        #     if not args.output_dir:
        #         print("Error: --output_dir is required for --single_model")
        #         return

        # model_name = os.path.splitext(os.path.basename(args.single_model))[0]
        # output_dir = os.path.join('G:\\My Drive\\output', model_name)

        # Hiển thị mô hình nếu được yêu cầu
        # if args.visualize:
        # creator.visualize_model(args.single_model, os.path.join(output_dir, "visualization.png"))
        # else:
        #     creator.process_model(args.single_model, output_dir)

    # else:
    #     print("No action specified. Use --create_json, --process_json, --process_directory, or --single_model")


    # Chạy chương trình
    main()