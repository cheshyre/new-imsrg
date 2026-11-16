#!/usr/bin/env python3
"""Convert a text or gzip Darmstadt me2j file to the binary `.me2j.bin` form.

Layout (design/19-mexj-formats.md 3.1, D57; imsrg++ ReadWrite.cc BINARYHEADERSIZE
255): a 255-byte header (the text header line, space padded or truncated to 255
bytes, so the tag of the text file survives), then every value of the text file
(after its first line) as a native little-endian IEEE-754 binary32 float, in the
same order. Values are parsed as Python floats (double) and rounded to float32,
which agrees with parsing the decimal text directly to float for the digits
counts of me2j files (<= 9 significant digits).

The input is read as a stream and written in chunks; do not rely on this for
files that must be bit-identical to a direct decimal-to-float parse beyond that.

Usage: me2j_text_to_bin.py INPUT[.gz] OUTPUT.me2j.bin
"""
import array
import gzip
import sys

HEADER_BYTES = 255
CHUNK = 1 << 20


def open_text(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb")
    return open(path, "rb")


def convert(src, dst):
    if sys.byteorder != "little":
        raise SystemExit("native little-endian host required")
    count = 0
    with open_text(src) as fin, open(dst, "wb") as fout:
        header = fin.readline().rstrip(b"\r\n")
        fout.write(header[:HEADER_BYTES].ljust(HEADER_BYTES, b" "))
        buf = array.array("f")
        for line in fin:
            buf.extend(float(tok) for tok in line.split())
            if len(buf) >= CHUNK:
                buf.tofile(fout)
                count += len(buf)
                buf = array.array("f")
        buf.tofile(fout)
        count += len(buf)
    return count


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    n = convert(argv[1], argv[2])
    print(f"wrote {n} floats ({HEADER_BYTES + 4 * n} bytes) to {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
