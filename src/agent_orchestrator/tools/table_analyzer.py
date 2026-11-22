from datetime import datetime
import re
from typing import Any

_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4})\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,4}[-/]\d{1,2}[-/]\d{2,4})\b"),
]

_PARSE_FORMATS = [
    "%d %B %Y", "%d %b %Y", "%d %B %y", "%d %b %y",
    "%B %d %Y", "%b %d %Y", "%B %d %y", "%b %d %y",
    "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
    "%Y-%m-%d", "%Y/%m/%d",
]

_STOPWORDS = {
    "what", "which", "when", "where", "how", "many", "much", "the", "and",
    "for", "from", "with", "that", "this", "convert", "total", "calculate",
    "final", "last", "first", "sum", "count", "between", "under", "about",
}


def _extract_date_formats(text: str) -> list[str]:
    """Return standard string representations for any date found in query text."""
    formats: list[str] = []
    seen: set[str] = set()
    for pattern in _DATE_PATTERNS:
        for match in pattern.findall(text):
            cleaned = re.sub(r"(\d{1,2})(?:st|nd|rd|th)\b", r"\1", match, flags=re.IGNORECASE)
            cleaned = " ".join(cleaned.replace(",", " ").split())
            dt = None
            for fmt in _PARSE_FORMATS:
                try:
                    dt = datetime.strptime(cleaned, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                continue
            for out_fmt in ("%d-%m-%y", "%d-%m-%Y", "%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"):
                val = dt.strftime(out_fmt)
                if val not in seen:
                    seen.add(val)
                    formats.append(val)
    return formats


def _parse_number(cell: str) -> float | None:
    cleaned = cell.replace(",", "").strip()
    if not cleaned or cleaned == "-":
        return None
    # Avoid treating pure date strings (e.g., "01-08-23" or "2025-01-15") as numeric amounts
    if re.fullmatch(r"\d{1,4}[-/]\d{1,2}[-/]\d{2,4}", cleaned):
        return None
    match = re.search(r"([-+]?\d+(?:\.\d+)?)", cleaned)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_markdown_table(content: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln.strip() for ln in content.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return [], []

    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    for line in lines[2:]:
        if set(line.replace("|", "").replace("-", "").strip()) == set():
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(cells)
    return headers, rows


def _infer_universal_currency(texts: list[str]) -> str | None:
    """Detect currency strictly from universal ISO codes or currency symbols."""
    combined = " ".join(texts)
    low = combined.lower()
    if "₹" in combined or re.search(r"\binr\b|\brs\.?\b", low):
        return "INR"
    if "$" in combined or re.search(r"\busd\b", low):
        return "USD"
    if "€" in combined or re.search(r"\beur\b", low):
        return "EUR"
    if "£" in combined or re.search(r"\bgbp\b", low):
        return "GBP"
    return None


def _extract_filter_keywords(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_-]{3,}", query.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def analyze_markdown_tables(
    evidences: list[dict[str, Any]], query: str
) -> dict[str, Any]:
    """
    Domain-agnostic Markdown table analyzer:
    - Parses any Markdown table regardless of column names
    - Filters rows by date (if present in query) or matches query keywords when selective
    - Deduplicates overlapping rows across chunks
    - Computes generic per-column stats (count, sum, min, max, first, last)
    """
    date_filters = _extract_date_formats(query)
    kw_filters = _extract_filter_keywords(query)

    all_rows: list[tuple[list[str], list[str]]] = []
    seen_row_keys: set[tuple[str, ...]] = set()
    table_texts: list[str] = []

    for ev in evidences:
        if ev.get("chunk_type") != "table":
            continue
        content = ev.get("content", "")
        headers, rows = _parse_markdown_table(content)
        if not headers or not rows:
            continue
        table_texts.append(content)

        for row in rows:
            row_key = (tuple(headers), tuple(row))
            if row_key in seen_row_keys:
                continue
            seen_row_keys.add(row_key)
            all_rows.append((headers, row))

    # 1. Apply date filter if the query specified a date
    if date_filters:
        filtered = [
            (h, r)
            for h, r in all_rows
            if any(df in " ".join(r) for df in date_filters)
        ]
    else:
        # 2. Otherwise check if specific entity/tier keywords narrow the rows cleanly
        kw_matched = [
            (h, r)
            for h, r in all_rows
            if any(kw in " ".join(r).lower() for kw in kw_filters)
        ]
        filtered = kw_matched if kw_matched else all_rows

    # Compute generic numeric summaries for every numeric column seen
    numeric_buckets: dict[str, list[float]] = {}
    matched_row_dicts: list[dict[str, str]] = []

    for headers, row in filtered:
        row_dict = dict(zip(headers, row, strict=True))
        matched_row_dicts.append(row_dict)
        for col_name, cell_val in row_dict.items():
            num = _parse_number(cell_val)
            if num is not None:
                numeric_buckets.setdefault(col_name, []).append(num)

    column_stats: dict[str, dict[str, float | int]] = {}
    for col_name, nums in numeric_buckets.items():
        column_stats[col_name] = {
            "count": len(nums),
            "sum": round(sum(nums), 4),
            "min": min(nums),
            "max": max(nums),
            "first": nums[0],
            "last": nums[-1],
        }

    # Pick a sensible primary numeric value (last value of the rightmost numeric column)
    primary_numeric_value: float | None = None
    primary_column: str | None = None
    if column_stats:
        primary_column = list(column_stats.keys())[-1]
        primary_numeric_value = float(column_stats[primary_column]["last"])

    return {
        "tool": "table_analyzer",
        "date_filters_applied": date_filters,
        "total_unique_rows_scanned": len(all_rows),
        "matched_row_count": len(matched_row_dicts),
        "column_stats": column_stats,
        "primary_column": primary_column,
        "primary_numeric_value": primary_numeric_value,
        "inferred_currency": _infer_universal_currency(table_texts + [query]),
        "matched_rows": matched_row_dicts[:25],
    }