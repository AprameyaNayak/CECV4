import json
from pathlib import Path
from typing import Any, Dict, List


# Automatically resolve paths relative to where reconstruction.py is saved
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "CM_DENTAL_NEET_JSON"
OUTPUT_DIR = BASE_DIR / "CM_COMBINED_TEXT"


def get_title(data: Dict[str, Any], fallback_stem: str) -> str:
    for key in ("title", "page_title"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return fallback_stem


def get_subtitles(data: Dict[str, Any]) -> List[str]:
    subtitles = data.get("subtitles")
    if isinstance(subtitles, list) and subtitles:
        return [str(s).strip() for s in subtitles if str(s).strip()]

    # Fallback 1: sections list
    sections = data.get("sections")
    if isinstance(sections, list) and sections:
        seen = []
        for sec in sections:
            if isinstance(sec, dict):
                h = str(sec.get("heading") or sec.get("heading_key") or "").strip()
                if h and h not in seen:
                    seen.append(h)
        if seen:
            return seen

    # Fallback 2: keys from text_content or content
    for dict_key in ("text_content", "content"):
        mapping = data.get(dict_key)
        if isinstance(mapping, dict) and mapping:
            return [str(k).strip() for k in mapping.keys() if str(k).strip()]

    return []


def get_subtitle_paragraphs(data: Dict[str, Any], subtitle: str) -> List[str]:
    for dict_key in ("text_content", "content"):
        mapping = data.get(dict_key)
        if isinstance(mapping, dict) and subtitle in mapping:
            raw_items = mapping[subtitle]
            if isinstance(raw_items, list):
                return [str(item).strip() for item in raw_items if str(item).strip()]
            if isinstance(raw_items, str) and raw_items.strip():
                return [raw_items.strip()]

    # Fallback: search sections
    sections = data.get("sections")
    if isinstance(sections, list):
        for sec in sections:
            if isinstance(sec, dict):
                heading = str(sec.get("heading") or sec.get("heading_key") or "").strip()
                if heading == subtitle:
                    raw_items = sec.get("content", [])
                    if isinstance(raw_items, list):
                        return [str(item).strip() for item in raw_items if str(item).strip()]

    return []


def format_table(rows: List[List[Any]]) -> str:
    cleaned_rows = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in rows
        if isinstance(row, list) and any(str(cell).strip() for cell in row)
    ]
    if not cleaned_rows:
        return ""

    col_count = max(len(r) for r in cleaned_rows)
    padded_rows = [r + [""] * (col_count - len(r)) for r in cleaned_rows]
    col_widths = [
        max(len(padded_rows[r_idx][c_idx]) for r_idx in range(len(padded_rows)))
        for c_idx in range(col_count)
    ]

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

    header_line = fmt_row(padded_rows[0])
    separator_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    body_lines = [fmt_row(r) for r in padded_rows[1:]]

    return "\n".join([header_line, separator_line, *body_lines])


def build_table_lookup(data: Dict[str, Any]) -> Dict[str, List[List[List[Any]]]]:
    lookup: Dict[str, List[List[List[Any]]]] = {}

    def add_table(key: Any, rows: Any) -> None:
        if not isinstance(key, str) or not key.strip():
            return
        if not isinstance(rows, list) or not rows:
            return
        norm_rows = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in rows
            if isinstance(row, list)
        ]
        if norm_rows:
            lookup.setdefault(key.strip(), []).append(norm_rows)

    # Primary source: table_content.tables
    table_content = data.get("table_content")
    if isinstance(table_content, dict):
        tables_list = table_content.get("tables")
        if isinstance(tables_list, list):
            for tbl in tables_list:
                if not isinstance(tbl, dict):
                    continue
                rows = tbl.get("table_values")
                if not rows and isinstance(tbl.get("rows"), list):
                    rows = [
                        r.get("cells", []) if isinstance(r, dict) else r
                        for r in tbl["rows"]
                    ]
                assoc = tbl.get("associated_section") or tbl.get("table_title")
                add_table(assoc, rows)

    # Secondary source: top-level tables dict
    top_tables = data.get("tables")
    if isinstance(top_tables, dict):
        for _, tbl in top_tables.items():
            if not isinstance(tbl, dict):
                continue
            rows = tbl.get("content")
            pos = tbl.get("position") or tbl.get("heading")
            if isinstance(pos, str) and pos.strip() in lookup:
                continue
            add_table(pos, rows)

    return lookup


def reconstruct_json_to_text(data: Dict[str, Any], fallback_stem: str = "document") -> str:
    title = get_title(data, fallback_stem)
    subtitles = get_subtitles(data)
    table_lookup = build_table_lookup(data)

    lines: List[str] = [title, ""]

    for subtitle in subtitles:
        lines.append(subtitle)

        paragraphs = get_subtitle_paragraphs(data, subtitle)
        for para in paragraphs:
            lines.append(para)

        associated_tables = table_lookup.get(subtitle, [])
        for table_rows in associated_tables:
            rendered = format_table(table_rows)
            if rendered:
                lines.append("")
                lines.append(rendered)

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"Error: Could not find folder '{INPUT_DIR}'")
        return

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No .json files found in '{INPUT_DIR}'")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_files = []

    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)

            text_output = reconstruct_json_to_text(data, fallback_stem=jf.stem)
            output_path = OUTPUT_DIR / f"{jf.stem}.txt"

            with output_path.open("w", encoding="utf-8") as f:
                f.write(text_output)

            success_count += 1
        except Exception as e:
            failed_files.append((jf.name, str(e)))

    print(f"Done! Reconstructed {success_count}/{len(json_files)} files into '{OUTPUT_DIR.name}/'.")
    if failed_files:
        print("\nFiles that failed to process:")
        for fname, err in failed_files:
            print(f"  - {fname}: {err}")


if __name__ == "__main__":
    main()