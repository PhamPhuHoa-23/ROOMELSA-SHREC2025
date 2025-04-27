import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import open3d as o3d


class PanoFormerDepthEstimator:
    """
    Panoformer for indoor 360 depth estimation
    """
    def __init__(self,
                 weights_path="tmp/panodepth/weights_pretrain",
                 device=None):
        # super().__init__()
        try:
            from .network.model import Panoformer as PanoFormerModel
        except ImportError:
            raise ImportError("Please install PanoFormerModel from PanoFormer")

        self.weights_path = os.path.expanduser(weights_path)
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using Device: {self.device}")

        self.model = PanoFormerModel()
        self.model.to(self.device)
        self.model.eval()

        self.load_weights()

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.max_depth_meters = 10.0

    def load_weights(self):
        try:
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"Weights path not found: {self.weights_path}")

            if os.path.isdir(self.weights_path):
                model_path = os.path.join(self.weights_path, "model.pth")
                # if not os.path.exists(model_path):
                #     pth_files = [f for f in os.listdir(self.weights_path) if f.endswith('.pth')]
                #     if not pth_files:
                #         raise FileNotFoundError(f"No .pth files found in {self.weights_path}")
                #     model_path = os.path.join(self.weights_path, pth_files[0])
                # else:
                #     model_path = self.weights_path

            print(f"Loading weights from {model_path}")

            state_dict = torch.load(model_path, map_location=self.device)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

            print("Weights loaded successfully")

        except Exception as e:
            print(f"Error: loading weights: {e}")
            raise

    def preprocess(self, image):
        try:
            if len(image.shape) !=  3 or image.shape[2] != 3:
                raise ValueError("Image must be RGB")

            resized_image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_CUBIC)
            tensor_image = self.to_tensor(resized_image)
            normalized_image = self.normalize(tensor_image)
            batched_image = normalized_image.unsqueeze(0)
            # batched_image = tensor_image.unsqueeze(0)

            return batched_image.to(self.device)
        except Exception as e:
            print(f"Error: preprocessing failed: {e}")
            raise

    def postprocess(self, model_output, original_size=None):
        try:
            pred_depth = model_output["pred_depth"]
            val_mask = ((pred_depth > 0) & (pred_depth < self.max_depth_meters) & ~torch.isnan(pred_depth))
            depth_map = pred_depth * val_mask.float()

            if original_size is not None:
                depth_map = depth_map.squeeze().cpu().numpy()
                print(depth_map.shape, type(depth_map))
                depth_map = cv2.resize(depth_map, (original_size[1], original_size[0]),
                                      interpolation=cv2.INTER_LINEAR)
            return depth_map
        except Exception as e:
            print(f"Error: postprocessing failed: {e}")
            raise

    def predict_depth(self, image):
        try:
            original_size = image.shape[:2]
            preprocessed_image = self.preprocess(image)

            with torch.no_grad():
                model_output = self.model(preprocessed_image)
                print(model_output)
                import matplotlib.pyplot as plt
                plt.imshow(model_output['pred_depth'].squeeze().cpu().numpy(), cmap='rainbow')
                plt.show()

            depth_map = self.postprocess(model_output, original_size)
            # depth_map = cv2.bilateralFilter(depth_map, 9, 75, 75)  # Edge-preserving smoothing

            return depth_map
        except Exception as e:
            print(f"Error: postprocessing failed: {e}")
            raise

    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_RAINBOW):
        """
        Visualize the depth map using a colormap.

        Args:
            depth_map (numpy.ndarray): Depth map as a 2D array.
            colormap (int, optional): OpenCV colormap to use. Default is cv2.COLORMAP_VIRIDIS.

        Returns:
            numpy.ndarray: Colorized depth map.
        """
        try:
            # Create a copy of the depth map to avoid modifying the original
            # depth_map = cv2.bilateralFilter(depth_map, 9, 75, 75)  # Edge-preserving smoothing
            vis_depth = depth_map.copy()

            # Find valid depth values (non-zero)
            valid_mask = (vis_depth > 0)

            if valid_mask.any():
                # Normalize only the valid depths for better visualization
                min_val = vis_depth[valid_mask].min()
                max_val = vis_depth[valid_mask].max()

                # Create a normalized version for visualization
                if max_val > min_val:
                    norm_depth = np.zeros_like(vis_depth)
                    norm_depth[valid_mask] = (vis_depth[valid_mask] - min_val) / (max_val - min_val)

                    # Scale to 0-255 range for colormapping
                    vis_depth = (norm_depth * 255).astype(np.uint8)


                else:
                    vis_depth = np.zeros_like(vis_depth, dtype=np.uint8)
            else:
                vis_depth = np.zeros_like(vis_depth, dtype=np.uint8)

            # Apply colormap
            colored_depth = cv2.applyColorMap(vis_depth, colormap)

            # Set invalid regions to black
            colored_depth[~valid_mask] = 0

            return colored_depth
        except Exception as e:
            print(f"Error visualizing depth map: {e}")
            raise

    def save_depth_map(self, depth_map, output_path, colormap=cv2.COLORMAP_RAINBOW, save_raw=True):
        """
        Save the depth map as an image and optionally as a raw numpy array.

        Args:
            depth_map (numpy.ndarray): Depth map as a 2D array.
            output_path (str): Path to save the depth map image.
            colormap (int, optional): OpenCV colormap to use. Default is cv2.COLORMAP_VIRIDIS.
            save_raw (bool, optional): Whether to save the raw depth map as a .npy file.
        """
        try:
            # Visualize the depth map
            colored_depth = self.visualize_depth(depth_map, colormap)

            # Save the image
            cv2.imwrite(output_path, colored_depth)
            print(f"Depth map visualization saved to: {output_path}")

            # Save raw depth map if requested
            if save_raw:
                raw_path = os.path.splitext(output_path)[0] + '.npy'
                np.save(raw_path, depth_map)
                print(f"Raw depth map saved to: {raw_path}")

        except Exception as e:
            print(f"Error saving depth map: {e}")
            raise

    @classmethod
    def to_point_cloud(cls,
                       depth_map,
                       rgb_image=None,
                       output_file=None,
                       visualize=False,
                       mask=None,
                       max_depth=10.0,
                       filter_outlier=None,
                       voxel_size=None
                       ):
        """

        :param depth_map: The depth map (hxw)
        :param rgb_image: RGB image (hxwx3)
        :param output_file: Path to save the point cloud
        :param mask: Binary mask for valid depth values (Support for instance mask in query)
        :param max_depth: Maximum depth value
        :param filter_outliers: Whether to filter out outliers
        :param voxel_size: Voxel size for downsampling depth map
        :return: open3d.geometry.PointCloud
        """
        h, w = depth_map.shape[:2]
        print(h, w)
        print(depth_map.max(), depth_map.min())
        if mask is None:
            mask = (depth_map > 0) & (depth_map < max_depth) & (~np.isnan(depth_map))

        points = []
        colors = []

        depth_map = depth_map / depth_map.max() * 1

        for i in range(h):
            for j in range(w):
                if mask[i, j]:
                    r = depth_map[i, j]
                    phi = (j / w) * 2 * np.pi
                    theta = (i / h) * np.pi
                    # r = 1

                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)

                    points.append([x, y, z])

                    if rgb_image is not None:
                        colors.append(rgb_image[i, j] / 255)

        points = np.array(points)
        colors = np.array(colors)
        # points = []
        # colors = []
        #
        # depth_map = depth_map / depth_map.max() * 1 # Standardization [0,1]
        #
        # j_grid, i_grid = np.meshgrid(range(w), range(h))
        #
        # phi = (j_grid / w) * 2 * np.pi
        # theta = (i_grid / h) * np.pi
        # r = depth_map
        #
        # x = r * np.sin(theta) * np.cos(phi)
        # y = r * np.sin(theta) * np.sin(phi)
        # z = r * np.cos(theta)
        #
        # valid_points = mask.flatten()
        # x_flat = x.flatten()[valid_points]
        # y_flat = y.flatten()[valid_points]
        # z_flat = z.flatten()[valid_points]
        #
        # points = np.column_stack((x_flat, y_flat, z_flat))
        #
        # if rgb_image is not None:
        #     if len(rgb_image.shape) == 3:
        #         colors = rgb_image.reshape(-1, rgb_image.shape[-1])[valid_points] / 255.0

        if len(points) == 0:
            raise ValueError("No points found")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # if filter_outlier is not None:
        #     pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)

        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        if output_file is not None:
            o3d.io.write_point_cloud(output_file, pcd)

        if visualize:
            cls.visualize_point_cloud(pcd)

        return pcd

    @classmethod
    def visualize_point_cloud(cls,
                              pcd,
                              window_name="ROOMELSA-SHREC-2025 Point Cloud"):
        """

        :param pcd: Point Cloud
        :param window_name: Window name
        :return:
        """
        o3d.visualization.draw_geometries([pcd], window_name=window_name)

    @classmethod
    def get_point_cloud_and_masked_point_cloud(cls,
                       depth_map,
                       image_path,
                       rgb_image=None,
                       output_file=None,
                       visualize=False,
                       max_depth=10.0,
                       filter_outlier=None,
                       voxel_size=None,
                       ):
        """
        Generate both regular point cloud and masked point cloud

        :param depth_map: The depth map (hxw)
        :param image_path: Path to the RGB image
        :param rgb_image: RGB image (hxwx3), will be loaded from image_path if None
        :param output_file: Path to save the point cloud (will be ignored as point clouds will be saved to the same directory as the image)
        :param visualize: Whether to visualize the point clouds
        :param max_depth: Maximum depth value
        :param filter_outlier: Whether to filter out outliers
        :param voxel_size: Voxel size for downsampling
        :return: Tuple of (regular point cloud, masked point cloud)
        """
        if rgb_image is None and image_path is not None:
            try:
                rgb_image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error loading image: {image_path}")

        if image_path is not None:
            dir_path = os.path.dirname(image_path)

            regular_output = os.path.join(dir_path, "room.ply")
            masked_output = os.path.join(dir_path, "masked_cloud.ply")

        else:
            if output_file is not None:
                dir_path = os.path.dirname(output_file)
                regular_output = os.path.join(dir_path, "room.ply")
                masked_output = os.path.join(dir_path, "masked_cloud.ply")

            else:
                regular_output = None
                masked_output = None

        standard_mask = (depth_map > 0) & (depth_map < max_depth) & (~np.isnan(depth_map))

        regular_pcd = cls.to_point_cloud(
            depth_map=depth_map,
            rgb_image=rgb_image,
            output_file=regular_output,
            visualize=visualize,
            mask=standard_mask,
            max_depth=max_depth,
            filter_outlier=filter_outlier,
            voxel_size=voxel_size,
        )

        mask_path = None
        if image_path is not None:
            mask_path = os.path.join(os.path.dirname(image_path), 'mask.npy')

        masked_pcd = None
        if mask_path is not None and os.path.exists(mask_path):
            instance_mask = np.load(mask_path)

            combined_mask = np.logical_and(instance_mask, standard_mask)

            masked_pcd = cls.to_point_cloud(
                depth_map=depth_map,
                rgb_image=rgb_image,
                output_file=masked_output,
                visualize=visualize,
                mask=combined_mask,
                max_depth=max_depth,
                filter_outlier=filter_outlier,
                voxel_size=voxel_size,
            )

        return regular_pcd, masked_pcd

