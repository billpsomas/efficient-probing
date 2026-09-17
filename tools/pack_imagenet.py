"""Convert an ImageFolder tree into the packed form of util/packed_dataset.py.

Motivation: ImageNet-1k is ~1.33M loose files, which is inode exhaustion on a
Lustre quota, while its ~150 GB is nothing against the block quota. Packing
turns each split into a handful of ~2 GiB shards of concatenated raw JPEG bytes
plus one npz index -- about 80 files for the whole dataset -- without changing a
single byte of any image.

Correctness is by construction plus verification, not trust:

  * enumeration goes through torchvision's ImageFolder itself, so sample order,
    labels and the class list are ImageFolder's exactly;
  * every byte written is the byte read; a per-shard sha256 is computed on the
    write path and stored in the index;
  * ``verify`` re-reads the shards and recomputes those hashes (catches
    truncation), re-enumerates the loose tree and asserts samples/classes
    equality (catches ordering), and byte-compares a seeded random sample of
    entries against the source files through the index (catches offset bugs).

Usage:
    python tools/pack_imagenet.py pack   --data_path DIR [--splits train val]
    python tools/pack_imagenet.py verify --data_path DIR [--splits train val]

Reads are prefetched by a thread pool (Lustre hides per-file latency well when
asked for many files at once); the shard writer stays strictly ordered.
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import os
import random
import sys
import time

import numpy as np
from torchvision import datasets

SHARD_BYTES = 2 << 30
VERIFY_SAMPLES = 8192


def enumerate_split(root):
    ds = datasets.ImageFolder(root)
    rel = [os.path.relpath(p, root) for p, _ in ds.samples]
    labels = [t for _, t in ds.samples]
    return ds.classes, rel, labels, [p for p, _ in ds.samples]


def pack_split(data_path, split, threads):
    root = os.path.join(data_path, split)
    index_path = root + ".index.npz"
    if os.path.exists(index_path):
        print("SKIP %s: %s exists" % (split, index_path))
        return
    classes, rel, labels, paths = enumerate_split(root)
    n = len(paths)
    print("%s: %d samples, %d classes" % (split, n, len(classes)))

    shard, offset, length = np.zeros(n, np.uint16), np.zeros(n, np.uint64), np.zeros(n, np.uint32)
    shard_names, shard_hashes = [], []
    out, sha, written, si = None, None, 0, -1
    t0 = time.time()

    def close_shard():
        if out is not None:
            out.close()
            shard_hashes.append(sha.hexdigest())

    def open_shard():
        nonlocal out, sha, written, si
        close_shard()
        si += 1
        name = "%s-%04d.pack" % (split, si)
        shard_names.append(name)
        out = open(os.path.join(data_path, name), "wb")
        sha = hashlib.sha256()
        written = 0

    open_shard()
    read = lambda p: open(p, "rb").read()
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        window = collections.deque()
        nxt = 0
        for i in range(n):
            while nxt < n and len(window) < 4 * threads:
                window.append(ex.submit(read, paths[nxt])); nxt += 1
            buf = window.popleft().result()
            if written + len(buf) > SHARD_BYTES and written > 0:
                open_shard()
            shard[i], offset[i], length[i] = si, written, len(buf)
            out.write(buf); sha.update(buf); written += len(buf)
            if (i + 1) % 50000 == 0:
                print("  %d/%d  (%.0f files/s)" % (i + 1, n, (i + 1) / (time.time() - t0)),
                      flush=True)
    close_shard()

    np.savez(index_path,
             classes=np.array(classes), paths=np.array(rel),
             label=np.array(labels, np.uint16),
             shard=shard, offset=offset, length=length,
             shards=np.array(shard_names), shard_sha256=np.array(shard_hashes))
    print("%s: wrote %d shards + index in %.0fs" % (split, len(shard_names), time.time() - t0))


def verify_split(data_path, split, threads):
    root = os.path.join(data_path, split)
    idx = np.load(root + ".index.npz", allow_pickle=False)
    n = len(idx["length"])
    fail = 0

    # 1. shard integrity: recompute every shard's sha256
    for name, want in zip(idx["shards"], idx["shard_sha256"]):
        h = hashlib.sha256()
        with open(os.path.join(data_path, str(name)), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        ok = h.hexdigest() == str(want)
        fail += not ok
        print("  shard %s sha256 %s" % (name, "OK" if ok else "MISMATCH"), flush=True)

    if not os.path.isdir(root):
        print("%s: loose tree gone, shard hashes only" % split)
        return fail == 0

    # 2. enumeration equality against a live ImageFolder walk
    classes, rel, labels, paths = enumerate_split(root)
    same = (list(map(str, idx["classes"])) == classes
            and list(map(str, idx["paths"])) == rel
            and idx["label"].tolist() == labels)
    fail += not same
    print("  enumeration (%d samples, %d classes): %s"
          % (n, len(classes), "OK" if same and n == len(rel) else "MISMATCH"))

    # 3. seeded random byte-compare through the index, plus every shard boundary
    take = set(random.Random(0).sample(range(n), min(VERIFY_SAMPLES, n)))
    take |= {0, n - 1} | {i for i in range(1, n) if idx["shard"][i] != idx["shard"][i - 1]}
    handles = {}
    for i in sorted(take):
        s = int(idx["shard"][i])
        if s not in handles:
            handles[s] = open(os.path.join(data_path, str(idx["shards"][s])), "rb")
        handles[s].seek(int(idx["offset"][i]))
        if handles[s].read(int(idx["length"][i])) != open(paths[i], "rb").read():
            fail += 1
            print("  BYTE MISMATCH at %d (%s)" % (i, rel[i]))
    print("  byte-compared %d entries: %s" % (len(take), "OK" if fail == 0 else "FAIL"))
    return fail == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "verify"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    ok = True
    for split in a.splits:
        if a.cmd == "pack":
            pack_split(a.data_path, split, a.threads)
        else:
            ok &= verify_split(a.data_path, split, a.threads)
    if a.cmd == "verify":
        print("VERIFY %s" % ("PASSED" if ok else "FAILED"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
