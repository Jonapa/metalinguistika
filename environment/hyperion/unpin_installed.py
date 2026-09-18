#!/usr/bin/env python3
"""
unpin_installed.py -- reconcile a pyproject.toml against an existing environment.

For every requirement declared in pyproject.toml, check whether that package is
already present in the target environment (conda- or pip-installed, both are
detected). If it is present, the version specifier is stripped so only the bare
package name (plus extras and environment markers) remains. That way
`pip install -e .` / `uv sync` will accept whatever is already there instead of
downgrading or upgrading it.

Requirements that are NOT installed keep their original pins untouched, unless
they are named with --force-unpin.

Manual unpinning
----------------
--force-unpin takes any number of package names and strips their pins whether
or not they show up in the environment scan. Useful for packages installed in a
way the scan cannot see (system packages, custom builds, editable checkouts
under a different dist name, CUDA wheels, ...). Names are matched with PEP 503
normalization, so "huggingface_hub", "huggingface-hub" and "HuggingFace.Hub"
are the same entry.

Usage
-----
    # inspect only, using the currently active interpreter's environment
    python unpin_installed.py pyproject.toml --dry-run

    # rewrite in place (a .bak copy is kept)
    python unpin_installed.py pyproject.toml

    # write somewhere else
    python unpin_installed.py pyproject.toml -o pyproject.local.toml

    # use a captured listing instead of the live environment
    conda list --export > env.txt        # or: pip freeze > env.txt
    python unpin_installed.py pyproject.toml --env-list env.txt

    # also unpin optional-dependencies / dependency-groups
    python unpin_installed.py pyproject.toml --sections all

    # unpin specific packages by hand, whatever the scan says
    python unpin_installed.py pyproject.toml --force-unpin torch torchvision numpy

Run it with the target environment ACTIVE (conda activate <env>) so that the
live scan looks at the right site-packages.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        sys.exit(
            "Need Python >=3.11 or `pip install tomli` to parse pyproject.toml."
        )


# --------------------------------------------------------------------------
# requirement parsing
# --------------------------------------------------------------------------

# name [extras] specifier [; marker]   -- good enough for pyproject dep strings
REQ_RE = re.compile(
    r"""^\s*
    (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)
    \s*
    (?P<extras>\[[^\]]*\])?
    \s*
    (?P<spec>[^;]*?)
    \s*
    (?:;\s*(?P<marker>.+?))?
    \s*$""",
    re.VERBOSE,
)


def normalize(name: str) -> str:
    """PEP 503 name normalization: huggingface_hub -> huggingface-hub."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def normalized_set(names) -> set[str]:
    """Normalize a list of hand-written package names, dropping blanks."""
    return {normalize(n) for n in names if n and n.strip()}


def parse_requirement(raw: str):
    """Return (name, extras, spec, marker) or None if unparseable."""
    if not raw or raw.lstrip().startswith("#"):
        return None
    # URL / VCS requirements: leave completely alone
    if "@" in raw.split(";")[0] or "://" in raw:
        return None
    m = REQ_RE.match(raw)
    if not m:
        return None
    return (
        m.group("name"),
        m.group("extras") or "",
        (m.group("spec") or "").strip(),
        (m.group("marker") or "").strip(),
    )


def rebuild(name: str, extras: str, marker: str) -> str:
    """Reassemble a requirement string with the version specifier removed."""
    out = f"{name}{extras}"
    if marker:
        out += f" ; {marker}"
    return out


# --------------------------------------------------------------------------
# environment discovery
# --------------------------------------------------------------------------


def installed_from_live_env() -> dict[str, str]:
    """Everything with importable metadata: pip installs and conda installs."""
    from importlib.metadata import distributions

    found: dict[str, str] = {}
    for dist in distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            name = None
        if not name:
            continue
        found.setdefault(normalize(name), dist.version or "unknown")
    return found


def installed_from_file(path: Path) -> dict[str, str]:
    """
    Parse a captured listing. Accepts:
      pip freeze          ->  numpy==2.1.3
      conda list --export ->  numpy=2.1.3=py312h...
      conda list          ->  numpy   2.1.3   py312h...   conda-forge
    """
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue
        if line.startswith("-e ") or "://" in line:
            continue

        if "==" in line:  # pip freeze
            name, _, ver = line.partition("==")
            ver = ver.split(";")[0].split()[0] if ver.strip() else "unknown"
        elif "=" in line and len(line.split()) == 1:  # conda --export
            parts = line.split("=")
            name, ver = parts[0], parts[1] if len(parts) > 1 else "unknown"
        else:  # whitespace table
            parts = line.split()
            if len(parts) < 2 or parts[0].lower() in {"name", "packages"}:
                continue
            name, ver = parts[0], parts[1]

        name = name.strip()
        if name:
            found.setdefault(normalize(name), ver.strip() or "unknown")
    return found


# --------------------------------------------------------------------------
# collecting declared requirements
# --------------------------------------------------------------------------


def collect(data: dict, sections: str):
    """Yield (origin, raw_requirement) pairs from the parsed pyproject."""
    project = data.get("project", {})

    for raw in project.get("dependencies", []) or []:
        yield "project.dependencies", raw

    if sections in {"all", "optional"}:
        for extra, reqs in (project.get("optional-dependencies") or {}).items():
            for raw in reqs or []:
                yield f"optional-dependencies.{extra}", raw

    if sections in {"all", "groups"}:
        for group, reqs in (data.get("dependency-groups") or {}).items():
            for raw in reqs or []:
                if isinstance(raw, str):
                    yield f"dependency-groups.{group}", raw


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

##############################
# ACTIVATE ENVIRONMENT FIRST!
##############################
# Only list
# python unpin_installed.py pyproject.toml --sections main --dry-run
# Modify file
# python unpin_installed.py pyproject.toml --sections main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strip version pins for packages already present in the environment."
    )
    ap.add_argument("pyproject", type=Path, help="path to pyproject.toml")
    ap.add_argument("-o", "--output", type=Path, help="write here instead of in place")
    ap.add_argument(
        "--env-list",
        type=Path,
        help="use a captured `pip freeze` / `conda list` file instead of the live env",
    )
    ap.add_argument(
        "--sections",
        choices=["main", "optional", "groups", "all"],
        default="main",
        help="which dependency tables to process (default: main)",
    )
    ap.add_argument(
        "--force-unpin",
        nargs="+",
        default=[],
        metavar="PKG",
        help="package names to unpin manually, regardless of the environment "
             "scan (space separated, e.g. --force-unpin torch numpy)",
    )
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument(
        "--no-backup", action="store_true", help="skip the .bak copy on in-place edits"
    )
    args = ap.parse_args()

    force_unpin = normalized_set(args.force_unpin)

    text = args.pyproject.read_text(encoding="utf-8")
    data = tomllib.loads(text)

    if args.env_list:
        installed = installed_from_file(args.env_list)
        source = f"file: {args.env_list}"
    else:
        installed = installed_from_live_env()
        source = f"live env: {sys.prefix}"

    rows = []
    replacements: dict[str, str] = {}
    seen_raw: set[str] = set()
    seen_names: set[str] = set()

    for origin, raw in collect(data, args.sections):
        parsed = parse_requirement(raw)
        if parsed is None:
            rows.append((origin, raw.strip(), "-", "skipped (url/vcs/unparsed)"))
            continue

        name, extras, spec, marker = parsed
        key = normalize(name)
        seen_names.add(key)
        have = installed.get(key)
        forced = key in force_unpin

        if have is None and not forced:
            rows.append((origin, name, "not installed", "kept pin"))
            continue

        if not spec:
            rows.append((origin, name, have or "not installed", "already unpinned"))
            continue

        why = " [forced]" if forced else ""
        new_raw = rebuild(name, extras, marker)
        rows.append(
            (origin, name, have or "not installed", f"unpinned{why} (was {spec})")
        )
        if raw not in seen_raw:
            replacements[raw] = new_raw
            seen_raw.add(raw)

    # ---- report -----------------------------------------------------------
    w_name = max([len(r[1]) for r in rows] + [7])
    w_ver = max([len(str(r[2])) for r in rows] + [9])
    print(f"Environment source: {source}")
    print(f"Packages visible in environment: {len(installed)}")
    print(f"Requirements inspected: {len(rows)}")
    if force_unpin:
        print(f"Forced unpins: {', '.join(sorted(force_unpin))}")
    print()
    print(f"{'PACKAGE'.ljust(w_name)}  {'INSTALLED'.ljust(w_ver)}  ACTION")
    print(f"{'-' * w_name}  {'-' * w_ver}  {'-' * 40}")
    for _origin, name, have, action in rows:
        print(f"{name.ljust(w_name)}  {str(have).ljust(w_ver)}  {action}")

    unmatched = force_unpin - seen_names
    if unmatched:
        print(
            "\nNote: these --force-unpin names matched no requirement in the "
            f"processed sections: {', '.join(sorted(unmatched))}"
        )

    changed = sum(1 for r in rows if r[3].startswith("unpinned"))
    kept = sum(1 for r in rows if r[3].startswith("kept pin"))
    print(f"\n{changed} unpinned, {kept} left pinned.")

    if not replacements:
        print("Nothing to rewrite.")
        return 0

    # ---- rewrite ----------------------------------------------------------
    new_text = text
    for old, new in replacements.items():
        # match the requirement inside its quotes so comments/formatting survive
        for q in ('"', "'"):
            needle = f"{q}{old}{q}"
            if needle in new_text:
                new_text = new_text.replace(needle, f"{q}{new}{q}")
                break
        else:
            print(f"  ! could not locate {old!r} verbatim; left unchanged")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    target = args.output or args.pyproject
    if target == args.pyproject and not args.no_backup:
        backup = args.pyproject.with_suffix(args.pyproject.suffix + ".bak")
        shutil.copy2(args.pyproject, backup)
        print(f"\nBackup: {backup}")

    target.write_text(new_text, encoding="utf-8")
    print(f"Wrote: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())