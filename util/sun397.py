import os
from pathlib import Path
from typing import Callable, Optional, Union, Tuple, List
import PIL.Image
from torchvision.datasets import SUN397 as TorchvisionSUN397

class SUN397(TorchvisionSUN397):
    """SUN397 Dataset with train-test split support based on external split files."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.split = split.lower()  # Store split info before calling superclass init
        super().__init__(root=root, transform=transform, target_transform=target_transform, download=download)

        # Filter images and labels based on the split files
        self._image_files, self._labels = self._load_split_files()

    def _load_split_files(self) -> Tuple[List[Path], List[int]]:
        """Load image paths and labels according to train/test split files."""
        
        if self.split == 'train':
            split_files = [self._data_dir / 'Training_01.txt']
        elif self.split == 'test':
            split_files = [self._data_dir / 'Testing_01.txt']
        else:
            print("Please define split for SUN397...")
        
        #split_files = sorted(
        #    self._data_dir.glob(f"{'Training' if self.split == 'train' else 'Testing'}_*.txt")
        #)

        image_files = []
        labels = []

        for split_file in split_files:
            with open(split_file) as f:
                for line in f:
                    rel_path = line.strip()  # e.g., "/a/abbey/sun_ajkqrqitspwywirx.jpg"
                    full_path = self._data_dir / rel_path[1:]  # Skip initial "/"
                    if full_path.exists():
                        image_files.append(full_path)
                        # Extract class name and handle potential discrepancies
                        class_name_parts = rel_path.split('/')[1:-1]  # Extract the path segments for class name
                        class_name = "/".join(class_name_parts[1:])  # Ignore 'a/' prefix

                        # Check if class_name matches with class_to_idx
                        if class_name in self.class_to_idx:
                            labels.append(self.class_to_idx[class_name])
                        else:
                            print(f"Class '{class_name}' not found in class_to_idx.")
        
        return image_files, labels

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[PIL.Image.Image, int]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label