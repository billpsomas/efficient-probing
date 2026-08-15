"""Fingerprint the CLI surface, and parse every command in the README.

The refactor is allowed to move code anywhere it likes but must not move a flag.
This dumps every option's name, default, type, action, nargs and choices, then
replays each README command through the parser (parse only -- main() is never
called, nothing touches the filesystem or the queue).

Usage:  python tools/inv_flags.py --out FILE [--readme FILE]
"""
import argparse
import json
import re
import shlex
import sys


def flag_surface(parser):
    out = {}
    for a in parser._actions:
        out[a.dest] = {
            "options": sorted(a.option_strings),
            "default": repr(a.default),
            "type": getattr(a.type, "__name__", repr(a.type)) if a.type else None,
            "action": type(a).__name__,
            "nargs": a.nargs,
            "choices": sorted(map(str, a.choices)) if a.choices else None,
            "required": a.required,
            "const": repr(a.const) if a.const is not None else None,
        }
    return out


def readme_commands(path):
    """Every `python3 ... main_linprobe.py ...` invocation in the README."""
    text = open(path).read()
    text = text.replace("\\\n", " ")  # join continuations
    cmds = []
    for line in text.split("\n"):
        if "main_linprobe.py" not in line:
            continue
        line = line.strip().lstrip("$").strip()
        i = line.index("main_linprobe.py")
        argstr = line[i + len("main_linprobe.py"):]
        argstr = re.sub(r"\s+", " ", argstr).strip()
        if argstr:
            cmds.append(argstr)
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--readme")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    import main_linprobe

    parser = main_linprobe.get_args_parser()
    result = {"flags": flag_surface(parser), "commands": {}}

    if args.readme:
        for cmd in readme_commands(args.readme):
            try:
                ns = parser.parse_args(shlex.split(cmd))
                # record the fully-resolved namespace: catches a changed default
                # that the flag dump alone would miss for flags the command omits
                result["commands"][cmd] = {
                    k: repr(v) for k, v in sorted(vars(ns).items())
                }
            except SystemExit as e:
                result["commands"][cmd] = {"__parse_error__": "SystemExit(%s)" % e.code}

    json.dump(result, open(args.out, "w"), indent=1, sort_keys=True)
    bad = sum(1 for v in result["commands"].values() if "__parse_error__" in v)
    print("wrote %s: %d flags, %d commands parsed, %d failed"
          % (args.out, len(result["flags"]), len(result["commands"]), bad))
    for c, v in result["commands"].items():
        if "__parse_error__" in v:
            print("  FAILED: %s" % c[:160])


if __name__ == "__main__":
    main()
