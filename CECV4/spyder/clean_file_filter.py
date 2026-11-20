from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TXT_DIR = BASE_DIR / "CM_COMBINED_TEXT"

# Set to True first to preview what will be deleted; set to False to actually delete.
DRY_RUN = False

# Also delete tiny/image-link-only files (e.g., under 150 bytes). Set to 0 to disable.
MAX_BYTES_IF_IMAGE_ONLY = 0

SUFFIXES_TO_DELETE = (
    "-2.txt",
    "avif.txt",
)


def should_delete(file_path: Path) -> bool:
    """Return True if a .txt file meets any deletion condition."""
    if not file_path.is_file() or file_path.suffix.lower() != ".txt":
        return False

    name_lower = file_path.name.lower()

    if any(name_lower.endswith(suffix.lower()) for suffix in SUFFIXES_TO_DELETE):
        return True

    if MAX_BYTES_IF_IMAGE_ONLY > 0 and file_path.stat().st_size <= MAX_BYTES_IF_IMAGE_ONLY:
        return True

    return False


def main():
    if not TXT_DIR.exists():
        print(f"Folder not found: {TXT_DIR}")
        return

    all_txt_files = sorted(TXT_DIR.glob("*.txt"))
    matched_files = [f for f in all_txt_files if should_delete(f)]

    print(f"Total .txt files scanned: {len(all_txt_files)}")
    print(f"Files matching delete conditions: {len(matched_files)}")
    print(f"Mode: {'DRY RUN (no files deleted)' if DRY_RUN else 'LIVE DELETE'}")
    print("-" * 60)

    for file_path in matched_files:
        size_bytes = file_path.stat().st_size
        if DRY_RUN:
            print(f"[WOULD DELETE] {file_path.name} ({size_bytes} bytes)")
        else:
            file_path.unlink()
            print(f"[DELETED]      {file_path.name} ({size_bytes} bytes)")

    print("-" * 60)
    remaining = len(list(TXT_DIR.glob("*.txt")))
    print(f"Remaining .txt files: {remaining}")


if __name__ == "__main__":
    main()