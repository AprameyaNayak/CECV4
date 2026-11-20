import json
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "CM_DENTAL_NEET_JSON"
DISCOVERED_URLS_FILE = BASE_DIR / "discovered_dental_neet_urls.txt"

ROOT_URL = "https://careermudhra.com/"
ALLOWED_HOSTS = {"careermudhra.com", "www.careermudhra.com"}

SITEMAP_CANDIDATES = [
    "https://careermudhra.com/sitemap_index.xml",
    "https://careermudhra.com/sitemap.xml",
    "https://careermudhra.com/wp-sitemap.xml",
]

# Discovery controls so crawling stays bounded and polite.
MAX_CRAWL_PAGES = 400
MAX_CRAWL_DEPTH = 3
MAX_URLS_TO_SCRAPE = 0  # 0 = scrape all matched URLs; set e.g. 50 for a cap

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
}

TIMEOUT = (10, 30)
REQUEST_DELAY_SECONDS = 1.0

# Target topics: BDS, MDS, dental college, NEET MDS, NEET UG
TARGET_PATTERNS = [
    re.compile(r"\bbds\b", re.IGNORECASE),
    re.compile(r"\bmds\b", re.IGNORECASE),
    re.compile(r"\bdental\s+colleges?\b", re.IGNORECASE),
    re.compile(r"\bneet[\s\-_]*mds\b", re.IGNORECASE),
    re.compile(r"\bneet[\s\-_]*ug\b", re.IGNORECASE),
]

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".mp4", ".mp3", ".avi", ".mov",
    ".css", ".js", ".json", ".xml", ".txt"
}

SKIP_PATH_SEGMENTS = {
    "/wp-admin", "/wp-login.php", "/feed", "/comments/feed",
    "/author/", "/tag/", "/category/", "/cart", "/checkout", "/my-account"
}

# Tags that should not contribute paragraph text.
# Anchor tags are deliberately NOT excluded:
# their visible text is meaningful webpage content.
EXCLUDED_TAGS = {
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "form",
    "button",
    "svg",
    "iframe",
    "img",
    "picture",
    "video",
    "audio"
}

# Common WordPress, Elementor and webpage structures
# that do not belong to the main article.
EXCLUDED_SELECTORS = [
    "nav",
    "header",
    "footer",
    "aside",
    ".sidebar",
    ".widget-area",
    ".comments-area",
    ".comment-respond",
    ".elementor-location-header",
    ".elementor-location-footer",
    ".breadcrumb",
    ".breadcrumbs",
    ".share-buttons",
    ".social-share",
    ".related-posts",
    ".advertisement",
    ".ads",
    "[role='navigation']",
    "[aria-hidden='true']"
]

# These selectors are checked in order.
CONTENT_SELECTORS = [
    ".entry-content",
    ".wp-block-post-content",
    ".post-content",
    ".elementor-widget-theme-post-content",
    "article",
    "main",
    ".elementor",
    "body"
]

# Table-of-contents containers are excluded separately.
TOC_SELECTORS = [
    ".ez-toc",
    ".lwptoc",
    ".toc",
    ".table-of-contents",
    ".elementor-toc",
    "[class*='table-of-contents']",
    "[id*='table-of-contents']"
]

HEADING_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6"
]

SUBTITLE_TAGS = [
    "h2", "h3", "h4", "h5", "h6"
]

CONTENT_TAGS = [
    "p",
    "li",
    "blockquote"
]


# ============================================================
# HTTP SESSION
# ============================================================

def create_session():
    """Create a reusable HTTP session with retries."""

    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# ============================================================
# DOMAIN & TOPIC FILTERING
# ============================================================

def normalize_careermudhra_url(raw_url):
    """Normalize URL and confirm it belongs strictly to CareerMudhra."""

    if not raw_url:
        return None

    cleaned = raw_url.strip()
    if not cleaned or cleaned.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None

    absolute_url = urljoin(ROOT_URL, cleaned)

    try:
        parsed = urlparse(absolute_url)
    except ValueError:
        return None

    if parsed.scheme.lower() not in {"http", "https"}:
        return None

    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        return None

    path = parsed.path or "/"
    lower_path = path.lower()

    if any(lower_path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return None

    if any(seg in lower_path for seg in SKIP_PATH_SEGMENTS):
        return None

    # Strip query strings and fragments so duplicate tracking URLs collapse.
    normalized_path = path if path == "/" else path.rstrip("/")
    return f"https://careermudhra.com{normalized_path}"


def matches_target_topics(text):
    """Return True if text mentions BDS, MDS, dental college(s), NEET MDS, or NEET UG."""

    if not text:
        return False

    return any(pattern.search(text) for pattern in TARGET_PATTERNS)


def is_relevant_dental_neet_url(url):
    """Check URL path/slug for target keywords."""

    parsed = urlparse(url)
    readable_url = unquote(f"{parsed.path} {parsed.fragment}").replace("-", " ").replace("_", " ")
    return matches_target_topics(readable_url)


def is_relevant_dental_neet_page(data):
    """Verify extracted page content matches BDS, MDS, dental college, NEET MDS, or NEET UG."""

    parts = [
        data.get("source_url", ""),
        data.get("final_url", ""),
        data.get("title", ""),
        " ".join(data.get("subtitles", [])),
    ]

    content_map = data.get("content", {})
    for items in content_map.values():
        parts.extend(items)

    return matches_target_topics(" ".join(parts))


def parse_sitemap_xml(xml_bytes):
    """Extract <loc> URLs from a sitemap or sitemap index XML payload."""

    urls = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return urls

    for elem in root.iter():
        if elem.tag.endswith("loc") and elem.text:
            urls.append(elem.text.strip())

    return urls


def discover_careermudhra_urls(session):
    """
    Discover CareerMudhra URLs via XML sitemaps and internal link traversal.
    Returns a deduplicated list of relevant BDS/MDS/dental college/NEET MDS/NEET UG URLs.
    """

    matched_urls = []
    seen_matched = set()

    def record_if_relevant(candidate):
        norm = normalize_careermudhra_url(candidate)
        if norm and is_relevant_dental_neet_url(norm) and norm not in seen_matched:
            seen_matched.add(norm)
            matched_urls.append(norm)

    # 1. Check sitemaps first.
    sitemap_queue = deque(SITEMAP_CANDIDATES)
    visited_sitemaps = set()

    while sitemap_queue:
        sitemap_url = sitemap_queue.popleft()
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)

        try:
            resp = session.get(sitemap_url, timeout=TIMEOUT)
            if resp.status_code != 200:
                continue
            locs = parse_sitemap_xml(resp.content)
            for loc in locs:
                if loc.lower().endswith(".xml"):
                    if loc not in visited_sitemaps:
                        sitemap_queue.append(loc)
                else:
                    record_if_relevant(loc)
        except requests.RequestException:
            continue

    # 2. Crawl internal CareerMudhra pages to catch links not in sitemaps.
    crawl_queue = deque([
        (normalize_careermudhra_url(ROOT_URL), 0),
        (normalize_careermudhra_url("https://careermudhra.com/dental-colleges/"), 1),
        (normalize_careermudhra_url("https://careermudhra.com/mds/"), 1),
        (normalize_careermudhra_url("https://careermudhra.com/bds/"), 1),
    ])
    visited_pages = set()

    while crawl_queue and len(visited_pages) < MAX_CRAWL_PAGES:
        current_url, depth = crawl_queue.popleft()
        if not current_url or current_url in visited_pages:
            continue
        visited_pages.add(current_url)
        record_if_relevant(current_url)

        if depth >= MAX_CRAWL_DEPTH:
            continue

        try:
            resp = session.get(current_url, timeout=TIMEOUT)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(resp.content, "lxml")
            for anchor in soup.find_all("a", href=True):
                next_url = normalize_careermudhra_url(anchor["href"])
                if not next_url or next_url in visited_pages:
                    continue

                record_if_relevant(next_url)

                # Prioritize traversing pages that look relevant or top-level hubs.
                if is_relevant_dental_neet_url(next_url) or depth + 1 < MAX_CRAWL_DEPTH:
                    crawl_queue.append((next_url, depth + 1))

        except requests.RequestException:
            continue

        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    return matched_urls


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def clean_text(element):
    """Extract normalized readable text from an element."""

    if element is None:
        return ""

    return " ".join(
        element.get_text(" ", strip=True).split()
    )


def create_filename(title, used_names):
    """Generate a filesystem-safe unique filename."""

    name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "-",
        title
    )

    name = re.sub(r"\s+", "-", name)
    name = name.strip(". -")
    name = name[:150]

    if not name:
        name = "webpage"

    original_name = name
    counter = 2

    while name.casefold() in used_names:

        suffix = f"-{counter}"

        name = (
            original_name[:150 - len(suffix)]
            + suffix
        )

        counter += 1

    used_names.add(name.casefold())

    return f"{name}.json"


def is_valid_http_url(url):
    """Validate URL format and restrict strictly to CareerMudhra."""

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.netloc)
            and (parsed.hostname or "").lower() in ALLOWED_HOSTS
        )

    except ValueError:
        return False


def is_inside_excluded_structure(element, stop_at=None):
    """Check whether an element belongs to excluded content."""

    current = element

    excluded_classes = {
        "sidebar",
        "widget-area",
        "comments-area",
        "comment-respond",
        "breadcrumb",
        "breadcrumbs",
        "share-buttons",
        "social-share",
        "related-posts",
        "advertisement",
        "ads"
    }

    while isinstance(current, Tag) and current is not stop_at:

        if current.name in EXCLUDED_TAGS:
            return True

        if current.get("aria-hidden") == "true":
            return True

        classes = set(current.get("class", []))

        if classes.intersection(excluded_classes):
            return True

        current = current.parent

    return False


def is_inside_toc(element, stop_at=None):
    """Check whether an element belongs to a table of contents."""

    current = element

    while isinstance(current, Tag) and current is not stop_at:

        classes = " ".join(current.get("class", [])).casefold()
        element_id = str(current.get("id", "")).casefold()

        if (
            "ez-toc" in classes
            or "lwptoc" in classes
            or "table-of-contents" in classes
            or "elementor-toc" in classes
            or "table-of-contents" in element_id
        ):
            return True

        current = current.parent

    return False


def clean_paragraph(element):
    """
    Extract readable text from a paragraph or list item.

    Keeps meaningful hyperlink text.
    Ignores images, scripts, styles and other unwanted
    nested elements.
    """

    if element is None:
        return ""

    text_parts = []

    for node in element.descendants:

        if not isinstance(node, NavigableString):
            continue

        parent = node.parent
        skip = False

        while isinstance(parent, Tag) and parent is not element:

            if parent.name in EXCLUDED_TAGS:
                skip = True
                break

            parent = parent.parent

        if not skip:
            text_parts.append(str(node))

    return " ".join(
        " ".join(text_parts).split()
    )


def remove_unwanted_elements(root):
    """Remove navigation and other unwanted structures."""

    for selector in EXCLUDED_SELECTORS + TOC_SELECTORS:

        for element in root.select(selector):
            element.decompose()


def is_valid_article_text(text):
    """Reject empty or navigation-like text."""

    if not text:
        return False

    normalized = text.strip()

    if not normalized:
        return False

    if normalized.casefold() in {
        "contents",
        "table of contents",
        "quick links"
    }:
        return False

    return True


# ============================================================
# CONTENT ROOT AND PAGE TITLE
# ============================================================

def find_content_root(soup):
    """Identify the main webpage content area."""

    for selector in CONTENT_SELECTORS:

        element = soup.select_one(selector)

        if element:
            return element

    return soup.body or soup


def get_page_title(soup, content_root, url):
    """Extract the article title."""

    h1 = content_root.find("h1")

    if h1:

        title = clean_text(h1)

        if title:
            return title

    if soup.title:

        title = clean_text(soup.title)

        if title:
            return title

    path = unquote(
        urlparse(url).path
    ).strip("/")

    if path:
        return path.split("/")[-1]

    return "homepage"


# ============================================================
# MAIN STRUCTURED EXTRACTION
# ============================================================

def extract_subtitles_and_content(content_root, page_title):
    """
    Extract subtitles and the text belonging to each subtitle.
    """

    subtitles = []
    content = {}

    heading_counts = {}

    current_subtitle = None

    elements = content_root.find_all(
        HEADING_TAGS + CONTENT_TAGS
    )

    for element in elements:

        if is_inside_excluded_structure(
            element,
            stop_at=content_root
        ):
            continue

        if is_inside_toc(
            element,
            stop_at=content_root
        ):
            continue

        if (
            element.name == "p"
            and element.find_parent("li")
        ):
            continue

        if (
            element.name == "li"
            and element.find_parent("li")
        ):
            continue

        if element.name in SUBTITLE_TAGS:

            heading_text = clean_text(element)

            if not is_valid_article_text(heading_text):
                current_subtitle = None
                continue

            heading_counts[heading_text] = (
                heading_counts.get(heading_text, 0) + 1
            )

            if heading_counts[heading_text] == 1:
                subtitle_key = heading_text
            else:
                subtitle_key = (
                    f"{heading_text} "
                    f"({heading_counts[heading_text]})"
                )

            subtitles.append(subtitle_key)

            content[subtitle_key] = []

            current_subtitle = subtitle_key

            continue

        if element.name == "h1":
            continue

        if element.name in CONTENT_TAGS:

            if current_subtitle is None:
                continue

            if element.find_parent("table"):
                continue

            text = clean_paragraph(element)

            if not is_valid_article_text(text):
                continue

            content[current_subtitle].append(text)

    return subtitles, content


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_rows(table):
    """
    Extract table content row-wise.
    """

    output_rows = []
    pending_spans = {}

    max_columns = 0

    rows = [
        tr for tr in table.find_all("tr")
        if tr.find_parent("table") is table
    ]

    for tr in rows:

        row = []
        column = 0

        while column in pending_spans:

            value, remaining = pending_spans[column]

            row.append(value)

            if remaining <= 1:
                del pending_spans[column]
            else:
                pending_spans[column] = (
                    value,
                    remaining - 1
                )

            column += 1

        cells = tr.find_all(
            ["th", "td"],
            recursive=False
        )

        for cell in cells:

            while column in pending_spans:

                value, remaining = pending_spans[column]

                row.append(value)

                if remaining <= 1:
                    del pending_spans[column]
                else:
                    pending_spans[column] = (
                        value,
                        remaining - 1
                    )

                column += 1

            value = clean_text(cell)

            try:
                colspan = max(
                    1,
                    int(cell.get("colspan", 1))
                )
            except (TypeError, ValueError):
                colspan = 1

            try:
                rowspan = max(
                    1,
                    int(cell.get("rowspan", 1))
                )
            except (TypeError, ValueError):
                rowspan = 1

            for offset in range(colspan):

                row.append(value)

                if rowspan > 1:

                    pending_spans[column + offset] = (
                        value,
                        rowspan - 1
                    )

            column += colspan

        if row:

            output_rows.append(row)

            max_columns = max(
                max_columns,
                len(row)
            )

    for row in output_rows:

        row.extend(
            [""] * (max_columns - len(row))
        )

    return output_rows


def find_table_position(table, content_root):
    """
    Find the most recent subtitle before the table.
    """

    current_subtitle = None

    elements = content_root.find_all(
        SUBTITLE_TAGS + ["table"]
    )

    for element in elements:

        if element is table:
            break

        if element.name in SUBTITLE_TAGS:

            if is_inside_excluded_structure(
                element,
                stop_at=content_root
            ):
                continue

            if is_inside_toc(
                element,
                stop_at=content_root
            ):
                continue

            heading = clean_text(element)

            if is_valid_article_text(heading):
                current_subtitle = heading

    return current_subtitle


def find_table_heading(table):
    """
    Find a table-specific heading.
    """

    caption = table.find(
        "caption",
        recursive=False
    )

    if caption:

        caption_text = clean_text(caption)

        if caption_text:
            return caption_text

    previous = table.find_previous_sibling()

    checked = 0

    while previous is not None and checked < 4:

        if isinstance(previous, Tag):

            if previous.name in HEADING_TAGS:

                heading_text = clean_text(previous)

                if heading_text:
                    return heading_text

            if previous.name == "p":

                paragraph_text = clean_text(previous)

                if paragraph_text and len(paragraph_text) < 180:
                    return paragraph_text

        previous = previous.find_previous_sibling()
        checked += 1

    return None


def extract_tables(content_root):
    """
    Extract all HTML tables.
    """

    tables = {}

    table_index = 0

    for table in content_root.find_all("table"):

        if is_inside_excluded_structure(
            table,
            stop_at=content_root
        ):
            continue

        if is_inside_toc(
            table,
            stop_at=content_root
        ):
            continue

        rows = extract_table_rows(table)

        if not rows:
            continue

        table_index += 1

        table_id = f"table{table_index}"

        table_heading = find_table_heading(table)

        table_position = find_table_position(
            table,
            content_root
        )

        tables[table_id] = {

            "heading": table_heading,

            "position": table_position,

            "content": rows

        }

    return tables


# ============================================================
# LEGACY COMPATIBILITY STRUCTURES
# ============================================================

def build_legacy_sections(subtitles, content):
    """
    Build the previous schema's sections list.
    """

    sections = []

    for index, subtitle in enumerate(subtitles, start=1):

        sections.append({

            "section_id": f"section_{index:04d}",

            "heading": subtitle,

            "heading_key": subtitle,

            "heading_level": None,

            "parent_heading": None,

            "content": content.get(subtitle, [])

        })

    return sections


def build_legacy_table_content(tables):
    """Convert the new table structure to the old structure."""

    legacy_tables = []

    for index, (table_id, table) in enumerate(
        tables.items(),
        start=1
    ):

        rows = table.get("content", [])

        legacy_tables.append({

            "table_id": table_id,

            "table_index": index,

            "table_title": (
                table.get("heading")
                or table.get("position")
                or f"Table {index}"
            ),

            "associated_section": table.get("position"),

            "table_header": False,

            "headers": [],

            "rows": [
                {"cells": row}
                for row in rows
            ],

            "table_values": rows

        })

    return {

        "table_count": len(legacy_tables),

        "tables": legacy_tables

    }


# ============================================================
# WEBPAGE PROCESSING
# ============================================================

def extract_webpage(url, session):
    """Download a webpage once and extract structured content."""

    response = session.get(
        url,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    # Enforce CareerMudhra domain even after redirects.
    final_host = (urlparse(response.url).hostname or "").lower()
    if final_host not in ALLOWED_HOSTS:
        raise ValueError(f"Redirected outside CareerMudhra: {response.url}")

    soup = BeautifulSoup(
        response.content,
        "lxml"
    )

    content_root_original = find_content_root(soup)

    content_root_soup = BeautifulSoup(
        str(content_root_original),
        "lxml"
    )

    content_root = (
        content_root_soup.body
        or content_root_soup
    )

    remove_unwanted_elements(content_root)

    page_title = get_page_title(
        soup,
        content_root,
        response.url
    )

    subtitles, content = extract_subtitles_and_content(
        content_root,
        page_title
    )

    tables = extract_tables(
        content_root
    )

    paragraph_count = sum(
        len(paragraphs)
        for paragraphs in content.values()
    )

    legacy_sections = build_legacy_sections(
        subtitles,
        content
    )

    legacy_table_content = build_legacy_table_content(
        tables
    )

    combined_data = {

        "schema_version": "3.0",

        "title": page_title,

        "subtitles": subtitles,

        "content": content,

        "tables": tables,

        "source_url": url,

        "final_url": response.url,

        "http_status": response.status_code,

        "fetched_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "page_title": page_title,

        "sections": legacy_sections,

        "table_content": legacy_table_content,

        "text_content": content,

        "extraction_stats": {

            "section_count": len(subtitles),

            "heading_count": len(subtitles),

            "paragraph_and_list_item_count": paragraph_count,

            "table_count": len(tables)

        }

    }

    return combined_data


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    session = create_session()

    try:

        print("Discovering CareerMudhra URLs for BDS, MDS, dental college, NEET MDS, NEET UG...")

        urls = discover_careermudhra_urls(session)

        if MAX_URLS_TO_SCRAPE and MAX_URLS_TO_SCRAPE > 0:
            urls = urls[:MAX_URLS_TO_SCRAPE]

        with open(DISCOVERED_URLS_FILE, "w", encoding="utf-8") as url_log:
            for u in urls:
                url_log.write(f"{u}\n")

        if not urls:
            print("No matching CareerMudhra URLs found.")
            return

        print(f"Matching URLs found: {len(urls)} (saved to {DISCOVERED_URLS_FILE.name})")
        print(f"Output folder: {OUTPUT_DIR}")
        print("-" * 60)

        successful = 0
        skipped_topic = 0
        failed = 0

        used_names = set()

        for number, url in enumerate(
            urls,
            start=1
        ):

            print(
                f"[{number}/{len(urls)}] Processing: {url}"
            )

            if not is_valid_http_url(url):

                print("  ERROR: Invalid or non-CareerMudhra URL. Skipping.")

                failed += 1

                continue

            try:

                data = extract_webpage(
                    url,
                    session
                )

                if not is_relevant_dental_neet_page(data):

                    print("  SKIPPED: Page content does not match target topics.")

                    skipped_topic += 1

                    continue

                filename = create_filename(
                    data["title"],
                    used_names
                )

                output_file = OUTPUT_DIR / filename

                with open(
                    output_file,
                    "w",
                    encoding="utf-8"
                ) as file:

                    json.dump(
                        data,
                        file,
                        indent=4,
                        ensure_ascii=False
                    )

                stats = data["extraction_stats"]

                print(f"  SUCCESS: {filename}")

                print(
                    f"  Subtitles: "
                    f"{stats['section_count']}"
                )

                print(
                    f"  Paragraphs/list items: "
                    f"{stats['paragraph_and_list_item_count']}"
                )

                print(
                    f"  Tables: "
                    f"{stats['table_count']}"
                )

                successful += 1

            except requests.RequestException as error:

                print(
                    f"  DOWNLOAD ERROR: {error}"
                )

                failed += 1

            except Exception as error:

                print(
                    f"  EXTRACTION ERROR: {error}"
                )

                failed += 1

            if (
                number < len(urls)
                and REQUEST_DELAY_SECONDS > 0
            ):

                time.sleep(
                    REQUEST_DELAY_SECONDS
                )

    finally:

        session.close()

    print("-" * 60)

    print("PROCESSING COMPLETE")

    print(f"Candidate URLs: {len(urls)}")

    print(f"Successful: {successful}")

    print(f"Skipped (topic filter): {skipped_topic}")

    print(f"Failed: {failed}")

    print(f"JSON output folder: {OUTPUT_DIR}")


if __name__ == "__main__":

    main()