"""Export the pinned commits of the rtc-integration repositories as tar files for docker build.

The integration branches are local, so the image is built from exports rather than clones:
context/<name>.tar holds the tree of the pinned commit, with every submodule that is checked out
in the worktree at the commit the superproject records (recursively). Uninitialized submodules,
such as OpenModelica's GUI clients, are left out, as the local builds leave them out.

Top-level folders listed in the fourth column of pins (comma-separated) are left out, such as
OpenModelica's GUI clients and testsuite, which the build does not use.

usage: python3 prepare_context.py [names...]   (default: all repos in pins)
"""

import io
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTEXT = HERE / "context"


def pins():
    """(name, worktree, commit, top-level folders to leave out) per line of the pins file"""
    for line in (HERE / "pins").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, worktree, commit, *rest = line.split()
            exclude = set(rest[0].split(",")) if rest else set()
            yield name, Path(worktree), commit, exclude


def git(repo, *args):
    # Read only: some submodules belong to the build container's user (OpenModelica's build
    # script hands SuiteSparse and lis to it), which git refuses without safe.directory
    command = ["git", "-c", "safe.directory=*", "-C", str(repo), *args]
    return subprocess.run(command, check=True, capture_output=True).stdout


def add_tree(out, repo, commit, prefix, exclude=()):
    """Add the tree of a commit, and its checked-out submodules, to an open tarfile."""
    archive = git(repo, "archive", "--format=tar", f"--prefix={prefix}", commit)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        for member in tree:
            if member.name == "pax_global_header":
                continue
            if member.name.split("/")[0] in exclude:
                continue
            out.addfile(member, tree.extractfile(member) if member.isfile() else None)
    for line in git(repo, "ls-tree", "-r", commit).decode().splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind != "commit" or path.split("/")[0] in exclude:
            continue
        sub = repo / path
        if (sub / ".git").exists():
            add_tree(out, sub, sha, f"{prefix}{path}/")


def main(names):
    CONTEXT.mkdir(exist_ok=True)
    manifest_file = CONTEXT / "MANIFEST"
    manifest = {}
    if manifest_file.exists():
        manifest = dict(line.split() for line in manifest_file.read_text().splitlines() if line)
    for name, worktree, commit, exclude in pins():
        if names and name not in names:
            continue
        target = CONTEXT / f"{name}.tar"
        with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as out:
            add_tree(out, worktree, commit, "", exclude)
        manifest[name] = commit
        print(f"{target.name}: {target.stat().st_size / 1e6:.0f} MB at {commit[:10]}")
    manifest_file.write_text("".join(f"{n} {c}\n" for n, c in sorted(manifest.items())))


if __name__ == "__main__":
    main(sys.argv[1:])
