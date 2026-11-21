"""Render the current Eye transfer overview from saved reconstruction results.

The former historical figure generator is preserved in the dated original backup.
Usage: python make_eye_????.py OUTPUT.png
"""
from pathlib import Path
import sys
from make_eye_transfer_reconstruction import render

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python make_eye_????.py OUTPUT.png')
    render(overview=Path(sys.argv[1]))
