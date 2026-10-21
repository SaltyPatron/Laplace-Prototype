"""Generate the canonical scalar cache through its Native owner.

The proving ground must not maintain a second coordinate/order generator.
Inputs are existing local UCD files; this command never acquires data.
"""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", default=os.environ.get("LAPLACE_NATIVE_TIER0"), required=not os.environ.get("LAPLACE_NATIVE_TIER0"))
    parser.add_argument("--ucd", default=os.environ.get("LAPLACE_UCD"), required=not os.environ.get("LAPLACE_UCD"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    subprocess.run([args.native, args.ucd, str(output)], check=True)
    if output.stat().st_size != 1112064 * 64:
        raise RuntimeError("Native did not produce the canonical 1,112,064 scalar records")


if __name__ == "__main__":
    main()
