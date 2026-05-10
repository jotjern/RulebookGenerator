#!/usr/bin/env python3
"""Delete generated ScenicRules rule branches."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO = Path(
    os.environ.get("SCENIC_RULES_PATH", "/Users/jogramnaestjernshaugen/ScenicRules")
)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result


def list_branches(repo: Path, pattern: str) -> list[str]:
    result = git(repo, "branch", "--format=%(refname:short)", "--list", pattern)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def current_branch(repo: Path) -> str:
    result = git(repo, "branch", "--show-current")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete generated ScenicRules local branches."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
        help=f"ScenicRules repo path. Default: {DEFAULT_REPO}",
    )
    parser.add_argument(
        "--pattern",
        default="rule/*",
        help="Local branch glob to delete. Default: rule/*",
    )
    parser.add_argument(
        "--base",
        default="main",
        help="Branch to checkout before deleting if currently on a generated branch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print branches that would be deleted without deleting them.",
    )
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    if not repo.exists():
        print(f"Repo does not exist: {repo}", file=sys.stderr)
        return 2

    git(repo, "rev-parse", "--is-inside-work-tree")
    branches = list_branches(repo, args.pattern)
    if not branches:
        print(f"No local branches matched {args.pattern!r} in {repo}")
        return 0

    print(f"Matched {len(branches)} generated branch(es):")
    for branch in branches:
        print(f"  {branch}")

    if args.dry_run:
        print("Dry run only; no branches deleted.")
        return 0

    current = current_branch(repo)
    if current in branches:
        print(f"Currently on {current}; checking out {args.base}.")
        git(repo, "checkout", args.base)

    git(repo, "branch", "-D", *branches)
    print(f"Deleted {len(branches)} branch(es).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
