"""A packed, map-style stand-in for torchvision's ImageFolder.

ImageNet-1k as loose JPEGs is ~1.33M files, which exhausts Lustre inode quotas
long before it touches the block quota. The packed form stores the SAME bytes in
a few dozen large shards:

    <root>.index.npz       offsets/lengths/labels/paths for every sample
    <root>-0000.pack       raw JPEG bytes, concatenated (~2 GiB each)
    <root>-0001.pack       ...

where <root> is the split directory the loose form would use (e.g.
``.../imagenet/train``). Built by tools/pack_imagenet.py, which enumerates the
source through ImageFolder itself, so ``samples`` here reproduces ImageFolder's
exact order, labels and class list -- DistributedSampler epochs, k-NN feature
order and eval_reimagenet's samples/pred alignment are all unchanged, and the
per-sample bytes are byte-identical, so decoded tensors are too.

Random access is a seek+read into an already-open shard, so a plain
DataLoader with workers behaves exactly as it does over loose files.
"""
import io
import os

import numpy as np
from PIL import Image
import torch.utils.data


class PackedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        root = root.rstrip("/")
        index = root + ".index.npz"
        if not os.path.exists(index):
            raise FileNotFoundError(index)
        idx = np.load(index, allow_pickle=False)
        self.root = root
        self.transform = transform
        self.classes = [str(c) for c in idx["classes"]]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self._shard_names = [str(s) for s in idx["shards"]]
        self._shard = idx["shard"]
        self._offset = idx["offset"]
        self._length = idx["length"]
        self.targets = idx["label"].tolist()
        # virtual paths: <root>/<synset>/<file>, matching what ImageFolder would
        # report for the loose tree -- consumers key on the last two components
        self.samples = [(os.path.join(root, str(p)), t)
                        for p, t in zip(idx["paths"], self.targets)]
        self.imgs = self.samples
        # file handles are opened lazily per process: a DataLoader worker must
        # never inherit its parent's descriptors (shared seek positions)
        self._handles = {}
        self._pid = None

    def _read(self, i):
        if self._pid != os.getpid():
            self._handles, self._pid = {}, os.getpid()
        s = int(self._shard[i])
        fh = self._handles.get(s)
        if fh is None:
            fh = self._handles[s] = open(
                os.path.join(os.path.dirname(self.root), self._shard_names[s]), "rb")
        fh.seek(int(self._offset[i]))
        return fh.read(int(self._length[i]))

    def __len__(self):
        return len(self._length)

    def __getitem__(self, i):
        buf = self._read(i)
        with Image.open(io.BytesIO(buf)) as img:   # same decode as ImageFolder's pil_loader
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]
