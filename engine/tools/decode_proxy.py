#!/usr/bin/env python3
"""Decode an image to little-endian f32 RGB for the native Linux CLIs (Pillow only)."""
import argparse
from array import array
import sys

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('output')
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    width, height = image.size
    if not (33 <= width <= 8192 and 33 <= height <= 8192):
        parser.error('image dimensions must be between 33 and 8192')
    # Each byte code is normalized and published as f32, as in the WIC reader.
    table = [code / 255.0 for code in range(256)]
    pixels = array('f', (table[code] for code in image.tobytes()))
    if sys.byteorder != 'little':
        pixels.byteswap()
    with open(args.output, 'wb') as handle:
        pixels.tofile(handle)
    print(width, height)


if __name__ == '__main__':
    main()
