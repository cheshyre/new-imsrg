"""``python -m imsrg run CONFIG.toml [--set key=value ...]`` runs a pipeline."""

import argparse
import sys

from . import _core, pipeline


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m imsrg", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the pipeline of a TOML config")
    r.add_argument("config")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    cfg = _core.load_config(file=args.config, set=args.set)
    res = pipeline.run(cfg)
    print(f"E_ref = {res.e_reference:.8f} MeV")
    if res.e0 is not None:
        print(f"E0    = {res.e0:.8f} MeV")
    print(f"output: {res.directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
