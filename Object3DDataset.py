import os
import json
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


class Object3DDataset(Dataset):
    """
    Dataset for loading 3D object models from the hierarchical directory structure.
    """

    def __init__(
            self,
            data_root: str,
            json_path: str = "object.json",
            model_type: str = "normalized",  # "normalized" or "raw"
            transform=None
    ):
        """
        Initialize the 3D object dataset.

        Args:
            data_root: Root directory of the dataset
            json_path: Path to the metadata JSON file relative to data_root
            model_type: Type of model to load ("normalized" or "raw")
            transform: Optional transform to apply to the data
        """
        self.data_root = data_root
        self.transform = transform
        self.model_type = model_type

        # Load metadata
        json_full_path = os.path.join(data_root, json_path)
        with open(json_full_path, 'r') as f:
            self.metadata = json.load(f)

        # Create flat list of items
        self.items = []
        for uuid1, uuid1_data in self.metadata.items():
            for uuid2, item_data in uuid1_data.items():
                # Determine which obj file to use based on model_type
                if model_type == "normalized":
                    obj_filename = "normalized_model.obj"
                else:
                    obj_filename = "raw_model.obj"

                # Create item entry
                item_path = os.path.join(uuid1, uuid2)
                obj_path = os.path.join(item_path, obj_filename)

                item = {
                    "uuid1": uuid1,
                    "uuid2": uuid2,
                    "path": item_path,
                    "obj_path": obj_path,
                    "mtl_path": os.path.join(item_path, "model.mtl"),
                    "texture_path": os.path.join(item_path, "texture.png"),
                    "image_path": os.path.join(item_path, "image.jpg"),
                    "description": item_data.get("description", "")
                }
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Dict containing item data with full file paths
        """
        item = self.items[idx]

        # Get full paths
        obj_path = os.path.join(self.data_root, item["obj_path"])
        mtl_path = os.path.join(self.data_root, item["mtl_path"])
        texture_path = os.path.join(self.data_root, item["texture_path"])
        image_path = os.path.join(self.data_root, item["image_path"])

        # Create result dictionary
        result = {
            "uuid1": item["uuid1"],
            "uuid2": item["uuid2"],
            "obj_path": obj_path,
            "mtl_path": mtl_path,
            "texture_path": texture_path,
            "image_path": image_path,
            "description": item["description"]
        }

        # Apply transform if specified
        if self.transform:
            result = self.transform(result)

        return result

    def get_item_by_uuid(self, uuid1: str, uuid2: str) -> Optional[Dict]:
        """
        Get an item by its UUIDs.

        Args:
            uuid1: First-level UUID
            uuid2: Second-level UUID

        Returns:
            Dict containing item data or None if not found
        """
        for idx, item in enumerate(self.items):
            if item["uuid1"] == uuid1 and item["uuid2"] == uuid2:
                return self.__getitem__(idx)
        return None