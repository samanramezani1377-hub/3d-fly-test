#!/usr/bin/env python3
"""Rebuild the original STEP assembly from the repository's text chunks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTS = ROOT / "assembly_parts"
OUTPUT = ROOT / "drone_parametric.step"
chunks = sorted(PARTS.glob("drone_parametric.step.part*"))
if len(chunks) != 6:
    raise SystemExit(f"Expected 6 assembly chunks, found {len(chunks)}")
with OUTPUT.open("w", encoding="ascii", newline="") as out:
    for chunk in chunks:
        out.write(chunk.read_text(encoding="ascii"))
print(f"Rebuilt {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
