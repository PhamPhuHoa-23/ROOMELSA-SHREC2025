import os
import argparse
import torch
import tqdm
from pathlib import Path
import time
import gc

# Import our custom classes
from Render.Object3DDataset import Object3DDataset
from Render.MultiViewRenderer import MultiViewRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-view renderings of 3D models")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the rendered images")
    parser.add_argument("--json_path", type=str, default="object.json",
                        help="Path to the metadata JSON file relative to data_root")
    parser.add_argument("--model_type", type=str, default="normalized", choices=["normalized", "raw"],
                        help="Type of model to use ('normalized' for normalized_model.obj, 'raw' for raw_model.obj)")
    parser.add_argument("--renderer", type=str, default="pytorch3d", choices=["pytorch3d", "open3d"],
                        help="Rendering engine to use")
    parser.add_argument("--image_size", type=int, default=600, help="Size of the rendered images (square)")
    parser.add_argument("--num_views", type=int, default=12, help="Number of viewpoints to render")
    parser.add_argument("--elevation", type=float, default=30.0, help="Camera elevation in degrees")
    parser.add_argument("--distance", type=float, default=2.0, help="Camera distance from the object")

    # GPU optimization parameters
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision for faster computation")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG quality (0-100, 0 for PNG). Using JPEG is faster and uses less disk space.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark before starting")

    # New renderer parameters
    parser.add_argument("--auto_distance", action="store_true", default=True,
                        help="Automatically calculate optimal camera distance based on model size")
    parser.add_argument("--distance_margin", type=float, default=1.2,
                        help="Margin multiplier for auto-distance calculation")
    parser.add_argument("--lighting_intensity", type=float, default=1.0,
                        help="Intensity of the lighting (affects ambient, diffuse and specular)")
    parser.add_argument("--background_color", type=str, default="1,1,1",
                        help="Background color (R,G,B) with values in [0, 1], comma-separated")

    # Model filtering options
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of models to process (for testing)")
    parser.add_argument("--uuid1", type=str, default=None, help="Process only models with this UUID1 (for testing)")
    parser.add_argument("--uuid2", type=str, default=None, help="Process only models with this UUID2 (for testing)")
    parser.add_argument("--uuid_list_file", type=str, default=None,
                        help="Path to a text file containing UUID1 values to process (one per line)")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Clear CUDA cache between models to prevent OOM errors")

    return parser.parse_args()

def run_benchmark(args, device):
    """Run benchmark to determine optimal batch size for the GPU"""
    print("Running benchmark to find optimal configuration...")

    # For our enhanced renderer, batch size is not a parameter
    # Instead, we'll focus on finding the optimal image size that won't cause OOM errors

    # Test with different image sizes
    print("Testing different image sizes...")
    test_sizes = [512, 600, 768, 1024]
    workable_sizes = []

    # Get the first model for testing
    dataset = Object3DDataset(
        data_root=args.data_root,
        json_path=args.json_path,
        model_type=args.model_type
    )

    if len(dataset) == 0:
        print("No models found, skipping benchmark")
        return args

    # Get the first model
    item_data = dataset[0]
    obj_path = item_data["obj_path"]
    texture_path = item_data["texture_path"] if os.path.exists(item_data["texture_path"]) else None

    if not os.path.exists(obj_path):
        print(f"Model file doesn't exist: {obj_path}")
        return args

    # Parse background color
    bg_color = [float(x) for x in args.background_color.split(",")]

    for size in test_sizes:
        try:
            print(f"Testing image size: {size}x{size}")
            # Create a renderer with this size
            renderer = MultiViewRenderer(
                device=device,
                image_size=size,
                custom_views=True,
                distance=args.distance,
                auto_distance=args.auto_distance,
                distance_margin=args.distance_margin,
                lighting_intensity=args.lighting_intensity,
                background_color=tuple(bg_color)
            )

            # Try to render one view
            mesh = renderer.load_mesh(obj_path, texture_path)
            renderer.render_mesh(mesh, 0.0)

            # If successful, add to workable sizes
            workable_sizes.append(size)
            print(f"✓ Image size {size}x{size} works")

            # Clean up GPU memory
            del renderer, mesh
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"✗ Image size {size}x{size} failed with error: {str(e)}")
            # Clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    if workable_sizes:
        # Choose the largest workable size that doesn't exceed the requested size
        best_size = max([s for s in workable_sizes if s <= args.image_size], default=min(workable_sizes))
        print(f"Best image size from benchmark: {best_size}x{best_size}")
        args.image_size = best_size
    else:
        print("No workable image sizes found. Defaulting to 224x224")
        args.image_size = 224

    return args


def save_with_quality(image, path, quality=95):
    """Save image with specified JPEG quality or as PNG if quality=0"""
    if quality > 0:
        # Save as JPEG with specified quality
        image.save(str(path) + '.jpg', 'JPEG', quality=quality)
        return str(path) + '.jpg'  # Đã sửa từ path.with_suffix('.jpg')
    else:
        # Save as PNG for lossless quality
        image.save(str(path) + '.png', 'PNG')  # Đã sửa từ path.with_suffix('.png')
        print("Ta ke ta ke")
        return str(path) + '.png'

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Set CUDA configuration for optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Better performance but less reproducible

        # Enable TF32 precision for Ampere GPUs (RTX 30xx series or newer)
        if torch.cuda.get_device_capability(0)[0] >= 8:
            print("Enabling TF32 precision for faster computation")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        print("CUDA not available, using CPU. Performance will be significantly reduced.")
        args.use_amp = False

    print(f"Loading dataset from {args.data_root}")
    dataset = Object3DDataset(
        data_root=args.data_root,
        json_path=args.json_path,
        model_type=args.model_type
    )

    # Filter dataset if UUIDs are specified
    uuid1_list = []
    if args.uuid_list_file:
        try:
            with open(args.uuid_list_file, 'r') as f:
                uuid1_list = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(uuid1_list)} UUID1 values from {args.uuid_list_file}")
        except Exception as e:
            print(f"Error loading UUID list file: {e}")
            print("Continuing without UUID filtering")

    # Filter dataset if UUIDs are specified via arguments or file
    if args.uuid1 or args.uuid2 or uuid1_list:
        filtered_items = []
        for item in dataset.items:
            # Check if the item matches the specified UUID1/UUID2 or is in the UUID list
            if (args.uuid1 is None or item["uuid1"] == args.uuid1) and \
                    (args.uuid2 is None or item["uuid2"] == args.uuid2) and \
                    (not uuid1_list or item["uuid1"] in uuid1_list):
                filtered_items.append(item)

        # Print stats about filtering
        old_count = len(dataset.items)
        dataset.items = filtered_items
        new_count = len(dataset.items)
        print(f"Filtered dataset from {old_count} to {new_count} models")

    print(f"Dataset contains {len(dataset)} models")

    # Apply limit if specified
    if args.limit is not None:
        dataset.items = dataset.items[:min(args.limit, len(dataset.items))]
        print(f"Limited to {len(dataset.items)} models")

    # Run benchmark if requested
    if args.benchmark and device.type == "cuda":
        args = run_benchmark(args, device)

    # Parse background color
    bg_color = [float(x) for x in args.background_color.split(",")]

    # Create the appropriate renderer
    if args.renderer == "pytorch3d":
        print(f"Using PyTorch3D renderer on {device}")
        print(f"Configuration: image_size={args.image_size}x{args.image_size}, "
              f"auto_distance={args.auto_distance}, lighting_intensity={args.lighting_intensity}")

        renderer = MultiViewRenderer(
            device=device,
            image_size=args.image_size,
            custom_views=True,  # Use default 12-view configuration
            elevation=args.elevation,
            distance=args.distance,
            auto_distance=args.auto_distance,
            distance_margin=args.distance_margin,
            lighting_intensity=args.lighting_intensity,
            # background_color=tuple(bg_color)
        )

    # Process each model
    print(f"Starting rendering {len(dataset.items)} models...")
    start_time = time.time()
    total_models = len(dataset.items)
    completed_models = 0

    # Setup for automatic mixed precision if enabled
    if args.use_amp and device.type == "cuda":
        print("Using automatic mixed precision (AMP)")
        amp_scaler = torch.cuda.amp.GradScaler()
    else:
        amp_scaler = None

    # Use tqdm to display progress
    for i, item_data in enumerate(tqdm.tqdm(dataset, desc="Rendering models")):
        uuid1 = item_data["uuid1"]
        uuid2 = item_data["uuid2"]
        obj_path = item_data["obj_path"]
        texture_path = item_data["texture_path"]

        # Skip if the model has already been rendered
        output_dir = Path(args.output_dir) / uuid1 / uuid2
        if output_dir.exists():
            # Check file pattern based on the format used
            file_ext = "jpg" if args.jpeg_quality > 0 else "png"

            # Count files
            pattern = f"*.{file_ext}"
            existing_files = list(output_dir.glob(pattern))

            # For the default 12-view configuration
            if len(existing_files) >= 12:
                completed_models += 1
                continue

        # Render and save the images
        try:
            if not os.path.exists(obj_path):
                print(f"Warning: Model file doesn't exist: {obj_path}")
                continue

            # Check if texture exists
            if not os.path.exists(texture_path):
                print(f"Warning: Texture file doesn't exist: {texture_path}")
                texture_path = None

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Load the mesh
            mesh = renderer.load_mesh(obj_path, texture_path)

            # Render images
            images = []
            for idx, (elev, azim) in enumerate(renderer.camera_positions):
                # Use AMP if enabled
                if amp_scaler:
                    with torch.cuda.amp.autocast():
                        image = renderer.render_mesh(mesh, azim, elev)
                else:
                    image = renderer.render_mesh(mesh, azim, elev)

                # Convert tensor to PIL Image for saving
                image_np = (image.cpu().numpy() * 255).astype('uint8')
                from PIL import Image
                pil_image = Image.fromarray(image_np)

                # Create filename based on elevation and azimuth
                elev_prefix = "n" if elev < 0 else "p" if elev > 0 else "z"
                elev_abs = abs(int(elev))
                azim_int = int(azim)
                image_filename = f"{elev_prefix}{elev_abs:02d}_{azim_int:03d}"

                # Save with specified quality
                image_path = output_dir / image_filename
                save_with_quality(pil_image, image_path, args.jpeg_quality)

            completed_models += 1

            # Print progress
            if (i + 1) % 10 == 0 or i == 0:
                elapsed_time = time.time() - start_time
                models_per_second = completed_models / elapsed_time if elapsed_time > 0 else 0
                estimated_time = (total_models - completed_models) / models_per_second if models_per_second > 0 else 0

                print(f"Progress: {completed_models}/{total_models} models " +
                      f"({models_per_second:.2f} models/sec, " +
                      f"ETA: {estimated_time / 60:.1f} min)")

            # Clear CUDA cache if requested
            if args.clear_cache and device.type == "cuda":
                del mesh
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"Error processing {uuid1}/{uuid2}: {e}")
            # Clear CUDA cache after error
            if device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    total_time = time.time() - start_time
    print(f"Finished rendering {completed_models}/{total_models} models in {total_time / 60:.1f} minutes!")
    print(
        f"Average processing time: {total_time / completed_models:.2f} seconds per model" if completed_models > 0 else "No models processed")


if __name__ == "__main__":
    main()