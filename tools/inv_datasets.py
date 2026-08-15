"""Check the dataset table calls exactly what the original if/elif chain called.

Replaces every dataset constructor with a spy, runs both the original chain (exec'd
out of the pre-refactor source) and build_datasets(), and compares the recorded
(ctor, args, kwargs) tuples. Transforms are compared by identity, so a table that
swapped train and val transforms would fail here rather than silently train on
centre crops.

No dataset is ever actually constructed, so this needs no data on disk.

Usage:  python tools/inv_datasets.py --orig ORIGINAL_MAIN.py
"""
import argparse
import sys


class Spy:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __call__(self, *a, **kw):
        self.log.append((self.name, a, {k: v for k, v in sorted(kw.items())}))
        return "<%s>" % self.name


def spies(log):
    names = ["ImageFolder", "Places365", "CIFAR100", "StanfordCars", "Food101",
             "FGVCAircraft", "DTD", "OxfordIIITPet", "STL10"]
    ds = type("datasets", (), {n: Spy(n, log) for n in names})
    return ds, Spy("SUN397", log), Spy("CUB200", log)


DATASETS = ["imagenet1k", "places365", "CIFAR100", "StanfordCars", "Food101",
            "FGVCAircraft", "SUN397", "DTD", "OxfordIIITPet", "CUB200", "stl10"]


class A:
    pass


def run_original(orig_src, name, tr, te):
    src = open(orig_src).read()
    start = src.index("    if args.dataset_name == 'imagenet1k':")
    end = src.index("    print(dataset_train)", start)
    body = "".join((l[4:] if l.startswith("    ") else l) + "\n" for l in src[start:end].split("\n"))
    log = []
    ds, sun, cub = spies(log)
    args = A()
    args.dataset_name, args.data_path = name, "/DATA"
    ns = {"datasets": ds, "SUN397": sun, "CUB200": cub, "os": __import__("os"),
          "args": args, "transform_train": tr, "transform_val": te}
    exec(body, ns)
    return log


def run_table(name, tr, te):
    import main_linprobe
    log = []
    ds, sun, cub = spies(log)
    saved = (main_linprobe.datasets, main_linprobe.SUN397, main_linprobe.CUB200)
    main_linprobe.datasets, main_linprobe.SUN397, main_linprobe.CUB200 = ds, sun, cub
    # rebuild the table against the spies
    specs = {}
    for k, (ctor, a, b) in main_linprobe.DATASET_SPECS.items():
        cname = getattr(ctor, "__name__", str(ctor))
        specs[k] = (Spy(cname, log), a, b)
    saved_specs = main_linprobe.DATASET_SPECS
    main_linprobe.DATASET_SPECS = specs
    args = A()
    args.dataset_name, args.data_path = name, "/DATA"
    try:
        main_linprobe.build_datasets(args, tr, te)
    finally:
        main_linprobe.datasets, main_linprobe.SUN397, main_linprobe.CUB200 = saved
        main_linprobe.DATASET_SPECS = saved_specs
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    a = ap.parse_args()
    sys.path.insert(0, ".")

    tr, te = object(), object()   # identity-comparable sentinels
    bad = 0
    for name in DATASETS:
        want = run_original(a.orig, name, tr, te)
        got = run_table(name, tr, te)
        # normalise: the original passes root positionally for stl10 only
        def norm(log):
            out = []
            for cname, args_, kwargs in log:
                kwargs = dict(kwargs)
                if args_:
                    kwargs["root"] = args_[0]
                out.append((cname, kwargs))
            return out
        w, g = norm(want), norm(got)
        ok = w == g
        bad += not ok
        print("%-14s %s" % (name, "ok" if ok else "MISMATCH"))
        if not ok:
            for x, y in zip(w, g):
                if x != y:
                    print("    was: %s" % (x,))
                    print("    now: %s" % (y,))
        # transforms must land on the right split
        for (cname, kw), split in zip(g, ("train", "val")):
            want_t = tr if split == "train" else te
            if kw.get("transform") is not want_t:
                print("    TRANSFORM SWAPPED on %s split" % split)
                bad += 1
    print("\n%s" % ("ALL 11 DATASETS IDENTICAL" if not bad else "%d PROBLEMS" % bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
