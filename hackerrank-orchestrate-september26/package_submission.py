#!/usr/bin/env python3
"""Deterministic, fail-closed submission packager for Buy or Wait? (Prompt 22).

Generates `code.zip` conforming to HackerRank Orchestrate competition specifications:
- Root files: README.md, AGENTS.md, problem_statement.md, .env.example, .gitignore, requirements.txt
- Directories: code/, dataset/, evaluation/usage_report.md
- Strict exclusions: .env, log.txt, .git, __pycache__, scratch, output.csv, audit scripts
- Deterministic: Normalized timestamps (2026-09-13 18:00:00) and sorted manifests
- Fail-closed: Scans every file for real secrets/credentials before writing
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sys
import zipfile
from typing import Dict, List, Set, Tuple

FIXED_TIMESTAMP: Tuple[int, int, int, int, int, int] = (2026, 9, 13, 18, 0, 0)

# Exact root files permitted in submission package
ALLOWED_ROOT_FILES: Set[str] = {
    "README.md",
    "AGENTS.md",
    "problem_statement.md",
    ".env.example",
    ".gitignore",
    "requirements.txt",
}

# Directories permitted for recursive inclusion
ALLOWED_DIRECTORIES: Set[str] = {
    "code",
    "dataset",
    "evaluation",
}

# Explicitly forbidden filename patterns / substrings
EXCLUDED_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\.env(?:\..+)?$"),         # .env, .env.local, etc. (except .env.example)
    re.compile(r"^log\.txt$", re.IGNORECASE),
    re.compile(r"^output\.csv$", re.IGNORECASE),  # Root output.csv (submitted separately)
    re.compile(r"^code\.zip$", re.IGNORECASE),
    re.compile(r"^package_submission\.py$"),
    re.compile(r"^generate_output\.py$"),
    re.compile(r"^audit_.*\.py$"),
    re.compile(r"^CLAUDE\.md$", re.IGNORECASE),
    re.compile(r".*\.py[co]$"),
    re.compile(r".*\.pyd$"),
    re.compile(r".*Thumbs\.db$"),
    re.compile(r".*\.DS_Store$"),
]

# Explicitly forbidden directory names anywhere in path
EXCLUDED_DIR_NAMES: Set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "scratch",
    ".gemini",
    ".vscode",
    ".idea",
    ".agents",
}

# Forbidden specific file paths
EXCLUDED_SPECIFIC_PATHS: Set[str] = {
    "code/evaluation/main.py",
    "code/evaluation/usage_report.md",
    "dataset/media/ocr_results.json",
}

# Secret detection patterns (fail-closed)
SUSPICIOUS_CONTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "NVIDIA API Key",
        re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),
    ),
    (
        "OpenAI API Key",
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
    ),
    (
        "GitHub Token",
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    ),
    (
        "Private Key",
        re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|PRIVATE) KEY-----"),
    ),
]

SAFE_KEY_PLACEHOLDERS: Set[str] = {
    "nvapi-your-nvidia-api-key-here",
    "nvapi-mock-valid-key",
    "nvapi-mock-key",
    "nvapi-mock-robust",
    "nvapi-mock-inj",
    "nvapi-SUPER-SECRET-TOKEN-12345",  # In test_nemotron redaction test
}


def is_safe_content(rel_path: str, content_bytes: bytes) -> Tuple[bool, str]:
    """Scan file content for credentials, failing closed if any real secret is detected."""
    # Only scan text files
    if rel_path.endswith((".png", ".jpg", ".jpeg", ".ico", ".bin")):
        return True, ""

    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return True, ""

    for label, pattern in SUSPICIOUS_CONTENT_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            if match in SAFE_KEY_PLACEHOLDERS or "mock" in match.lower() or "dummy" in match.lower():
                continue
            return False, f"Detected potential {label} in '{rel_path}': {match[:8]}...[REDACTED]"

    return True, ""


def collect_submission_files(repo_root: Path) -> List[Path]:
    """Collect all files eligible for submission, enforcing strict allowlist & denylist."""
    collected: List[Path] = []

    for root, dirs, files in os.walk(repo_root):
        rel_dir = os.path.relpath(root, repo_root)
        parts = () if rel_dir == "." else Path(rel_dir).parts

        # Prune excluded directories
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIR_NAMES
            and not d.startswith(".")
            and (len(parts) == 0 and d in ALLOWED_DIRECTORIES or len(parts) > 0)
        ]

        if len(parts) == 0:
            # Root directory
            for f in files:
                if f in ALLOWED_ROOT_FILES:
                    collected.append(repo_root / f)
        else:
            top_dir = parts[0]
            if top_dir not in ALLOWED_DIRECTORIES:
                continue

            for f in files:
                file_path = Path(root) / f
                rel_path = file_path.relative_to(repo_root).as_posix()

                # Check specific exclusions
                if rel_path in EXCLUDED_SPECIFIC_PATHS:
                    continue

                # Check regex exclusions
                if any(p.match(f) for p in EXCLUDED_PATTERNS):
                    continue

                if f == ".env" or (f.startswith(".env.") and f != ".env.example"):
                    continue

                collected.append(file_path)

    return sorted(collected, key=lambda p: p.relative_to(repo_root).as_posix())


def build_submission_zip(repo_root: Path, output_zip_path: Path) -> Dict[str, object]:
    """Build deterministic code.zip package with strict verification."""
    files_to_pack = collect_submission_files(repo_root)

    # Pre-check all files for secrets before opening ZIP
    manifest: List[Tuple[str, int, str]] = []
    total_uncompressed = 0

    for file_path in files_to_pack:
        rel_path = file_path.relative_to(repo_root).as_posix()
        data = file_path.read_bytes()

        safe, reason = is_safe_content(rel_path, data)
        if not safe:
            raise ValueError(f"SECURITY VIOLATION: {reason}")

        sha = hashlib.sha256(data).hexdigest()
        manifest.append((rel_path, len(data), sha))
        total_uncompressed += len(data)

    # Write deterministic ZIP
    if output_zip_path.exists():
        output_zip_path.unlink()

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_pack:
            rel_path = file_path.relative_to(repo_root).as_posix()
            data = file_path.read_bytes()

            zinfo = zipfile.ZipInfo(filename=rel_path, date_time=FIXED_TIMESTAMP)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.external_attr = 0o644 << 16  # Standard file permissions

            zf.writestr(zinfo, data)

    # Post-packaging verification
    zip_bytes = output_zip_path.read_bytes()
    zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    zip_size = len(zip_bytes)

    with zipfile.ZipFile(output_zip_path, "r") as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Generated ZIP corrupted at {corrupt}")

        namelist = zf.namelist()

    # Mandatory inclusions verification
    required_inclusions = [
        "code/main.py",
        "evaluation/usage_report.md",
        "dataset/requests.csv",
        "dataset/financial_profiles.csv",
        "dataset/financial_events.csv",
        "dataset/images.csv",
        "dataset/media/images/image_01.png",
        "dataset/media/images/image_16.png",
        "README.md",
        "AGENTS.md",
        "problem_statement.md",
        "requirements.txt",
        ".env.example",
        ".gitignore",
    ]

    for req in required_inclusions:
        if req not in namelist:
            raise ValueError(f"Required file missing from ZIP: {req}")

    # Mandatory exclusions verification
    forbidden_inclusions = [
        ".env",
        "log.txt",
        "output.csv",
        "code.zip",
        "package_submission.py",
        "dataset/media/ocr_results.json",
        "code/evaluation/main.py",
        "code/evaluation/usage_report.md",
    ]

    for forb in forbidden_inclusions:
        if forb in namelist:
            raise ValueError(f"Forbidden file found inside ZIP: {forb}")

    for name in namelist:
        if "__pycache__" in name or name == ".git" or name.startswith(".git/") or "/.git/" in name or name.startswith("scratch/"):
            raise ValueError(f"Forbidden directory found inside ZIP: {name}")

    return {
        "zip_path": str(output_zip_path),
        "zip_sha256": zip_sha256,
        "zip_size_bytes": zip_size,
        "total_files": len(namelist),
        "total_uncompressed_bytes": total_uncompressed,
        "manifest": namelist,
    }


def main():
    repo_root = Path(__file__).resolve().parent
    output_zip = repo_root / "code.zip"

    print("=====================================================")
    print(" deterministic submission packager (Prompt 22)      ")
    print("=====================================================")
    print(f"Repository Root : {repo_root}")
    print(f"Target Output   : {output_zip}")

    try:
        summary = build_submission_zip(repo_root, output_zip)
        print("\n[SUCCESS] Submission ZIP created successfully.")
        print(f"  File count        : {summary['total_files']}")
        print(f"  Uncompressed size : {summary['total_uncompressed_bytes']:,} bytes")
        print(f"  Compressed size   : {summary['zip_size_bytes']:,} bytes")
        print(f"  SHA-256           : {summary['zip_sha256']}")
        print("\nIncluded root files:")
        for name in summary["manifest"]:
            if "/" not in name:
                print(f"  - {name}")
        print("\nDirectories included:")
        top_dirs = sorted(set(n.split("/")[0] for n in summary["manifest"] if "/" in n))
        for d in top_dirs:
            count = sum(1 for n in summary["manifest"] if n.startswith(d + "/"))
            print(f"  - {d}/ ({count} files)")
        return 0
    except Exception as e:
        print(f"\n[FAILED] Packaging aborted: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
