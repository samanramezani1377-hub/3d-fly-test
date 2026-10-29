"""Build a native shared library from the same C++ core used by the browser WASM build."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PINNED_REVISION = "a31864f999418531775980b879979ce4c57e791d"


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=root / ".cache" / "skysim-upstream")
    parser.add_argument("--output", type=Path, default=root / "training" / "libskysim_core.so")
    args = parser.parse_args(argv)
    upstream = args.upstream_dir.resolve()
    output = args.output.resolve()
    upstream.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not (upstream / ".git").exists():
        run(["git", "clone", "https://github.com/vishwagw/Sky-Sim-drone-simulator-3.0.git", upstream])
    run(["git", "fetch", "origin", PINNED_REVISION], cwd=upstream)
    run(["git", "checkout", "--detach", PINNED_REVISION], cwd=upstream)
    run(["git", "reset", "--hard", PINNED_REVISION], cwd=upstream)
    run(["git", "clean", "-fdx"], cwd=upstream)

    # Apply exactly the same project config/battery compatibility patch as the
    # browser build. This changes only the temporary upstream checkout.
    run([
        "python", str(root / "scripts" / "prepare_site.py"),
        "--upstream", upstream,
        "--commit-sha", "native-training",
        "--upstream-ref", PINNED_REVISION,
    ], cwd=root)

    run([
        "g++", "-std=c++20", "-O3", "-fPIC", "-shared",
        "-I", upstream / "include",
        upstream / "web" / "core" / "drone_core.cpp",
        upstream / "web" / "core" / "capi.cpp",
        "-o", output,
    ])
    print(f"Built native SkySim physics core: {output}")
    print(f"Source revision: {PINNED_REVISION}")
    print("The native library and browser WASM build share the same DroneCore C++ sources and config patch.")


if __name__ == "__main__":
    main()
