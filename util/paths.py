"""Where this project keeps things, resolved once so no script hardcodes a path.

Every location is an environment variable with a repo-relative default, so a
fresh clone works with no setup while a cluster install can point each one
somewhere else (typically big, purgeable storage) by exporting the variable --
see the EP_* names below.

    from util import paths
    open(os.path.join(paths.RELEASE, "manifest.json"))

Deliberately free of any absolute path: this file is public, and the machine it
was written on is not.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(var, *default):
    """Environment variable `var`, else REPO/<default...>."""
    return os.environ.get(var) or os.path.join(REPO, *default)


ROOT = _p("EP_ROOT")                              # project root, if set up
OUTPUTS = _p("EP_OUTPUTS", "outputs")             # one directory per training run
DATASETS = _p("EP_DATASETS", "data")
IMAGENET = _p("EP_IMAGENET", DATASETS, "imagenet")
REIMAGENET = _p("EP_REIMAGENET", DATASETS, "reimagenet")
BACKBONES = _p("EP_BACKBONES", "pretrained_models")
RELEASE = _p("EP_RELEASE", "release")             # exported heads + manifest.json
PREDS = _p("EP_PREDS", "preds")
WORK = _p("EP_WORK", "work")                      # scratch space for smoke tests etc.
NOTES = _p("EP_NOTES", "notes")                   # maintainer notes, not in git
LOGS = _p("EP_LOGS", "logs")
TORCH_HOME = os.environ.get("EP_TORCH_HOME") or os.environ.get("TORCH_HOME") or \
    os.path.expanduser("~/.cache/torch")
HUB_CHECKPOINTS = os.path.join(TORCH_HOME, "hub", "checkpoints")
