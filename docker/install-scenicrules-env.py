#!/usr/bin/env python3
"""Install ScenicRules into the active virtualenv.

The upstream ScenicRules pyproject currently cannot be installed directly:
it depends on a Scenic fork that declares antlr4-python3-runtime ~=4.11, while
ScenicRules, Hydra, and OmegaConf need antlr4-python3-runtime 4.9.*. Install
the Scenic dependency set first, then force the runtime back to 4.9.3 before
installing Hydra.

Keep that workaround here so the Dockerfile does not need to duplicate every
ScenicRules dependency by hand.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


ANTLR_RUNTIME = "antlr4-python3-runtime==4.9.3"


def pip_install(*args: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", *args]
    )


def main() -> int:
    scenicrules_src = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/scenicrules-src")
    pyproject_path = scenicrules_src / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        project = tomllib.load(f)["project"]

    dependencies = project.get("dependencies", [])
    normal_deps: list[str] = []
    for dep in dependencies:
        normalized = dep.strip().lower()
        if normalized.startswith("antlr4-python3-runtime"):
            continue
        if normalized.startswith("hydra-core"):
            continue
        normal_deps.append(dep)

    pip_install("--upgrade", "pip", "setuptools", "wheel")
    pip_install("--no-deps", str(scenicrules_src))
    pip_install(*normal_deps)
    pip_install(ANTLR_RUNTIME)
    pip_install("hydra-core~=1.3.2")
    pip_install("openai", "gitpython")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
