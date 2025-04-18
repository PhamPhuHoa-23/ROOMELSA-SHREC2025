import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union

# PyTorch3D imports
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
    TexturesVertex,
)


class MultiViewRenderer:
    """
    Class for rendering 3D models from multiple viewpoints using PyTorch3D.
    """

    def __init__(
            self,
            device: torch.device = None,
            image_size: int = 224,
            custom_views: bool = True,
            num_views: int = 12,
            elevation: float = 30.0,
            distance: float = 2.0,
            auto_distance: bool = True,  # New parameter to enable auto-distance calculation
            distance_margin: float = 1.2,  # Margin multiplier for auto-distance
            background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
            lighting_intensity: float = 1.0  # New parameter to control lighting intensity
    ):
        """
        Initialize the renderer.

        Args:
            device: Device to use for rendering (CPU or CUDA)
            image_size: Size of the rendered images (square)
            custom_views: Whether to use custom 12-view configuration (3 elevations × 4 azimuths)
            num_views: Number of viewpoints to render (only used if custom_views=False)
            elevation: Camera elevation in degrees (only used if custom_views=False)
            distance: Camera distance from the object (used as default or minimum if auto_distance=True)
            auto_distance: Whether to automatically calculate optimal camera distance based on model size
            distance_margin: Margin multiplier for auto-distance calculation (higher = more space around model)
            background_color: Background color (R, G, B) with values in [0, 1]
            lighting_intensity: Intensity of the lighting (affects ambient, diffuse and specular)
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.custom_views = custom_views
        self.num_views = num_views
        self.elevation = elevation
        self.distance = distance
        self.auto_distance = auto_distance
        self.distance_margin = distance_margin
        self.background_color = background_color
        self.lighting_intensity = lighting_intensity
        self.current_distance = distance  # Will be updated if auto_distance is True

        if custom_views:
            # Define the 12 specific views: 3 elevations × 4 azimuths
            # Create a list of (elevation, azimuth) pairs
            elevations = [-30.0, 0.0, 30.0]
            azimuths = [45.0, 135.0, 225.0, 315.0]

            self.camera_positions = []
            for elev in elevations:
                for azim in azimuths:
                    self.camera_positions.append((elev, azim))

            self.num_views = len(self.camera_positions)  # Should be 12
        else:
            # Calculate azimuth angles for different viewpoints (evenly spaced around 360°)
            self.azimuths = torch.linspace(0, 360, num_views + 1)[:-1]

        # Set up the renderer
        self._setup_renderer()

    def _setup_renderer(self):
        """Set up the PyTorch3D renderer with default parameters."""
        # Rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=100000  # Increase to handle complex models
        )

        # Calculate light intensities based on lighting_intensity parameter
        ambient = 0.5 * self.lighting_intensity
        diffuse = 0.7 * self.lighting_intensity
        specular = 0.3 * self.lighting_intensity

        # Create a renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=None,  # Cameras will be set for each render
                lights=PointLights(
                    device=self.device,
                    location=[[0.0, 0.0, 3.0]],
                    ambient_color=((ambient, ambient, ambient),),
                    diffuse_color=((diffuse, diffuse, diffuse),),
                    specular_color=((specular, specular, specular),)
                )
            )
        )

    def load_mesh(self, obj_path: str, texture_path: Optional[str] = None) -> Meshes:
        """
        Load a 3D mesh from an .obj file.

        Args:
            obj_path: Path to the .obj file
            texture_path: Optional path to the texture image

        Returns:
            PyTorch3D Meshes object
        """
        try:
            # Load the mesh
            mesh = load_objs_as_meshes(
                [obj_path],
                device=self.device,
                # load_textures=True,
                # create_texture_atlas=True,  # Create texture atlas for better quality
            )
            mesh = self._center_and_scale_mesh(mesh)

            # Calculate optimal distance if auto_distance is enabled
            if self.auto_distance:
                self.current_distance = self._calculate_optimal_distance(mesh)
            else:
                self.current_distance = self.distance
                
            return mesh


            # Convert faces to the right format for PyTorch3D
            faces_idx = faces.verts_idx
            texture_path = texture_path.replace("""\\\\""", "/").replace("""\\""", "/").replace("\\", "\\\\")
            print(texture_path)

            # Create a Textures object
            if texture_path is not None and os.path.exists(texture_path) and hasattr(aux,
                                                                                     'verts_uvs') and aux.verts_uvs is not None:
                # Load texture image
                try:
                    texture_image = Image.open(texture_path).convert("RGB")

                    # Resize texture to power of 2 dimensions for better GPU performance if needed
                    width, height = texture_image.size
                    if not (self._is_power_of_two(width) and self._is_power_of_two(height)):
                        new_width = self._next_power_of_two(width)
                        new_height = self._next_power_of_two(height)
                        texture_image = texture_image.resize((new_width, new_height), Image.LANCZOS)

                    texture_image = torch.from_numpy(np.array(texture_image) / 255.0).to(
                        dtype=torch.float32, device=self.device
                    )

                    # Create UV texture
                    textures = TexturesUV(
                        maps=[texture_image],
                        faces_uvs=[aux.face_uvs_idx] if hasattr(aux,
                                                                'face_uvs_idx') and aux.face_uvs_idx is not None else None,
                        verts_uvs=[aux.verts_uvs]
                    )
                    print(f"Successfully loaded texture from {texture_path}")
                except Exception as e:
                    print(f"Error loading texture: {e}")
                    # Create improved fallback texture with gradient
                    verts_rgb = self._create_gradient_texture(verts)
                    textures = TexturesVertex(verts_features=verts_rgb)
            else:
                # Create improved default texture with subtle color variation
                verts_rgb = self._create_gradient_texture(verts)
                textures = TexturesVertex(verts_features=verts_rgb)

            # Create a Meshes object
            mesh = Meshes(
                verts=[verts],
                faces=[faces_idx],
                textures=textures
            )

            # Center and scale the mesh
            mesh = self._center_and_scale_mesh(mesh)

            # Calculate optimal distance if auto_distance is enabled
            if self.auto_distance:
                self.current_distance = self._calculate_optimal_distance(mesh)
                print(f"Auto-calculated camera distance: {self.current_distance}")
            else:
                self.current_distance = self.distance

            return mesh

        except Exception as e:
            print(f"Error loading mesh: {e}")
            raise

    def _is_power_of_two(self, n):
        """Check if a number is a power of two."""
        return n > 0 and (n & (n - 1)) == 0

    def _next_power_of_two(self, n):
        """Get the next power of two greater than or equal to n."""
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1
        return n

    def _create_gradient_texture(self, verts):
        """
        Create a gradient texture based on vertex positions for better visual quality.

        Args:
            verts: Mesh vertices

        Returns:
            Tensor with RGB values for each vertex
        """
        # Get normalized height (y-coordinate) of each vertex
        heights = verts[:, 1].clone()
        min_height, max_height = heights.min(), heights.max()
        height_range = max_height - min_height

        if height_range > 0:
            # Normalize heights to [0, 1]
            heights = (heights - min_height) / height_range

            # Create gradient colors (bluish at bottom to whitish at top)
            colors = torch.zeros_like(verts)
            colors[:, 0] = 0.7 + heights * 0.3  # R: 0.7-1.0
            colors[:, 1] = 0.7 + heights * 0.3  # G: 0.7-1.0
            colors[:, 2] = 0.9  # B: constant 0.9

            return colors.unsqueeze(0)  # Add batch dimension
        else:
            # Fallback to uniform light gray if no height variation
            return torch.ones_like(verts).unsqueeze(0) * 0.8

    def _calculate_optimal_distance(self, mesh: Meshes) -> float:
        """
        Calculate the optimal camera distance to ensure the entire model is visible.

        Args:
            mesh: PyTorch3D Meshes object

        Returns:
            Optimal camera distance
        """
        # Get vertices
        verts = mesh.verts_packed()

        # Calculate the bounding sphere radius
        # Since model is centered at origin and scaled to unit sphere in _center_and_scale_mesh,
        # the radius should be close to 1.0, but we calculate it explicitly to be sure
        radius = torch.norm(verts, dim=1).max().item()

        # Calculate field of view in radians (default is 60 degrees for FoVPerspectiveCameras)
        fov_radians = torch.tensor(60.0).to(self.device) * np.pi / 180.0

        # Calculate minimum distance needed to see the whole object
        # Using the formula: distance = radius / tan(fov/2)
        min_distance = radius / torch.tan(fov_radians / 2.0).item()

        # Apply margin and ensure at least the minimum specified distance
        optimal_distance = max(min_distance * self.distance_margin, self.distance)

        return optimal_distance

    def _center_and_scale_mesh(self, mesh: Meshes) -> Meshes:
        """
        Center and scale the mesh to fit within a unit sphere.

        Args:
            mesh: PyTorch3D Meshes object

        Returns:
            Centered and scaled mesh
        """
        # Get vertices
        verts = mesh.verts_packed()

        # Compute center and scale
        center = verts.mean(dim=0)
        verts = verts - center

        # Scale to unit sphere
        scale = verts.abs().max()
        verts = verts / scale

        # Update mesh vertices
        new_verts_list = [verts]
        new_mesh = Meshes(
            verts=new_verts_list,
            faces=mesh.faces_list(),
            textures=mesh.textures
        )

        return new_mesh

    def render_mesh(self, mesh: Meshes, azimuth: float, elevation: float = None) -> torch.Tensor:
        """
        Render a mesh from a specific viewpoint.

        Args:
            mesh: PyTorch3D Meshes object
            azimuth: Camera azimuth angle in degrees
            elevation: Camera elevation angle in degrees (optional)

        Returns:
            Rendered image as tensor of shape (H, W, 3)
        """
        # Use provided elevation or default
        elev = elevation if elevation is not None else self.elevation

        # Create the camera for this viewpoint
        R, T = look_at_view_transform(
            dist=self.current_distance,  # Use calculated optimal distance
            elev=elev,
            azim=azimuth
        )
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # Calculate light intensities based on lighting_intensity parameter
        ambient = 0.5 * self.lighting_intensity
        diffuse = 0.7 * self.lighting_intensity
        specular = 0.3 * self.lighting_intensity

        # Render the mesh
        images = self.renderer(
            meshes_world=mesh,
            cameras=cameras,
            lights=PointLights(
                device=self.device,
                location=[[0.0, 0.0, self.current_distance * 1.5]],  # Position light relative to camera distance
                ambient_color=((ambient, ambient, ambient),),
                diffuse_color=((diffuse, diffuse, diffuse),),
                specular_color=((specular, specular, specular),)
            ),
            bg_color=self.background_color
        )

        # Return the rendered image
        return images[0, ..., :3]  # (H, W, 3)

    def render_multiview(self, obj_path: str, texture_path: Optional[str] = None) -> List[torch.Tensor]:
        """
        Render a 3D model from multiple viewpoints.

        Args:
            obj_path: Path to the .obj file
            texture_path: Optional path to the texture image

        Returns:
            List of rendered images as tensors of shape (H, W, 3)
        """
        # Load the mesh
        mesh = self.load_mesh(obj_path, texture_path)

        # Render from multiple viewpoints
        images = []

        if self.custom_views:
            # Use the predefined camera positions (elevation, azimuth pairs)
            for elev, azimuth in self.camera_positions:
                # Use the render_mesh method for consistency
                image = self.render_mesh(mesh, azimuth, elev)
                images.append(image)
        else:
            # Use evenly spaced azimuth angles at a fixed elevation
            for azimuth in self.azimuths:
                image = self.render_mesh(mesh, azimuth.item())
                images.append(image)

        return images

    def save_rendered_images(
            self,
            obj_path: str,
            output_dir: str,
            uuid1: str,
            uuid2: str,
            texture_path: Optional[str] = None,
            create_dirs: bool = True
    ) -> List[str]:
        """
        Render a 3D model from multiple viewpoints and save the images.

        Args:
            obj_path: Path to the .obj file
            output_dir: Directory to save the rendered images
            uuid1: First-level UUID for directory structure
            uuid2: Second-level UUID for directory structure
            texture_path: Optional path to the texture image
            create_dirs: Whether to create directories if they don't exist

        Returns:
            List of paths to the saved images
        """
        # Create output directory structure
        item_dir = os.path.join(output_dir, uuid1, uuid2)
        if create_dirs and not os.path.exists(item_dir):
            os.makedirs(item_dir, exist_ok=True)

        # Render the images
        rendered_images = self.render_multiview(obj_path, texture_path)

        # Save the images
        image_paths = []
        for i, image in enumerate(rendered_images):
            # Convert tensor to numpy image (0-255)
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)

            # Create output path with a naming convention that indicates elevation and azimuth
            if self.custom_views:
                elev, azim = self.camera_positions[i]
                # Format: elev_azim (e.g., n30_090 for elevation -30, azimuth 90)
                elev_prefix = "n" if elev < 0 else "p" if elev > 0 else "z"  # n=negative, p=positive, z=zero
                elev_abs = abs(int(elev))
                azim_int = int(azim)
                image_filename = f"{elev_prefix}{elev_abs:02d}_{azim_int:03d}.png"
            else:
                image_filename = f"view_{i:02d}.png"

            image_path = os.path.join(item_dir, image_filename)

            # Save the image
            Image.fromarray(image_np).save(image_path)
            image_paths.append(image_path)

        return image_paths