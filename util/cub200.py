import os
from PIL import Image
from torch.utils.data import Dataset

class CUB200(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = self._load_split_paths()  # Initialize image paths based on split
        self.class_to_idx = {name: i for i, name in enumerate(sorted(set(os.path.dirname(p).split('/')[-1] for p in self.image_paths)))}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

    def _load_split_paths(self):
        # File paths for the dataset
        split_file = os.path.join(self.root, "train_test_split.txt")
        image_paths_file = os.path.join(self.root, "images.txt")
        image_paths = {}
        split_ids = []

        # Read all image paths
        with open(image_paths_file, "r") as f:
            for line in f:
                image_id, path = line.strip().split()
                image_paths[int(image_id)] = os.path.join(self.root, "images", path)

        # Read split information and filter based on specified split
        with open(split_file, "r") as f:
            for line in f:
                image_id, is_train = line.strip().split()
                is_train = int(is_train)
                if (self.split == "train" and is_train) or (self.split == "test" and not is_train):
                    split_ids.append(int(image_id))

        # Return the paths corresponding to the specified split
        return [image_paths[id] for id in split_ids]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[os.path.dirname(img_path).split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:  # Apply target_transform if provided
            label = self.target_transform(label)
        return image, label