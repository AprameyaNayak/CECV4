import hashlib
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TXT_DIR = BASE_DIR / "CM_COMBINED_TEXT"
DB_PATH = BASE_DIR / "dental_fts.sqlite"

# Filler/question words that add noise across broad admission/counselling pages.
# Keep domain nouns/acronyms (bds, mds, neet, ug, pg, fees, seat, seats, cutoff, eligibility, etc.)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "belong",
    "by", "can", "could", "did", "do", "does", "doing", "done", "for",
    "from", "get", "give", "got", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "know",
    "let", "like", "make", "many", "may", "me", "might", "more", "most",
    "much", "must", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "please", "same", "she", "should", "so", "some",
    "such", "tell", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "us", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "completed", "qualified", "scored", "available", "specific",
    "previous", "year", "years", "compulsory", "internship", "category",
    "domicile", "general", "sc", "st", "obc", "ews", "nri", "marks",
    "percentage", "rank"
}

TOP_CHUNKS_FOR_FILE_SCORE = 3


def parse_txt_into_chunks(file_path: Path):
    """
    Split a reconstructed .txt file into section-level chunks.
    Returns tuples of: (filename, page_title, section_title, body_text)
    """
    raw = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []

    lines = raw.splitlines()
    page_title = lines[0].strip() if lines else file_path.stem
    rest_text = "\n".join(lines[1:]).strip()

    if not rest_text:
        return [(file_path.name, page_title, page_title, raw)]

    blocks = [b.strip() for b in re.split(r"\n\s*\n+", rest_text) if b.strip()]

    chunks = []
    current_section = "Overview"

    for block in blocks:
        block_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not block_lines:
            continue

        first_line = block_lines[0]
        is_table_line = first_line.startswith("|") or set(first_line) <= {"|", "-", ":", " "}
        if len(block_lines) == 1 and not is_table_line and len(first_line) <= 120:
            current_section = first_line
            continue

        chunks.append((
            file_path.name,
            page_title,
            current_section,
            block,
        ))

    if not chunks:
        chunks.append((file_path.name, page_title, page_title, raw))

    return chunks


def _normalized_content_hash(file_path: Path) -> str:
    """Hash normalized text so duplicate/variant files collapse during indexing."""
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    normalized = " ".join(raw.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_fts_index(txt_dir: Path = TXT_DIR, db_path: Path = DB_PATH):
    """Build (or rebuild) the SQLite FTS5 database from unique .txt files."""
    if not txt_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {txt_dir}")

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE VIRTUAL TABLE sections_fts USING fts5(
            filename UNINDEXED,
            page_title,
            section_title,
            body_text,
            tokenize = 'porter unicode61'
        )
    """)

    all_rows = []
    seen_hashes = set()
    txt_files = sorted(txt_dir.glob("*.txt"))
    indexed_files = 0
    skipped_duplicates = 0

    for txt_file in txt_files:
        content_hash = _normalized_content_hash(txt_file)
        if content_hash in seen_hashes:
            skipped_duplicates += 1
            continue
        seen_hashes.add(content_hash)
        indexed_files += 1
        all_rows.extend(parse_txt_into_chunks(txt_file))

    cur.executemany(
        """
        INSERT INTO sections_fts (filename, page_title, section_title, body_text)
        VALUES (?, ?, ?, ?)
        """,
        all_rows,
    )

    conn.commit()
    conn.close()
    print(
        f"Indexed {len(all_rows)} chunks from {indexed_files} unique files "
        f"(skipped {skipped_duplicates} duplicates) into {db_path.name}"
    )


def _build_fts_query(user_query: str) -> str:
    """
    Convert a natural-language question into a clean FTS5 MATCH expression.
    Filters out noisy stopwords while keeping specific names, codes, and topic words.
    """
    raw_tokens = re.findall(r"[A-Za-z0-9]+", user_query)
    if not raw_tokens:
        return ""

    # Deduplicate while preserving order
    seen = set()
    filtered = []
    fallback = []

    for tok in raw_tokens:
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)

        if len(low) >= 2:
            fallback.append(tok)

        if low not in STOPWORDS and (len(low) >= 2 or low.isdigit()):
            filtered.append(tok)

    tokens_to_use = filtered if filtered else fallback
    if not tokens_to_use:
        return ""

    quoted_tokens = [f'"{token}"' for token in tokens_to_use]
    return " OR ".join(quoted_tokens)


def search_top_files(query: str, db_path: Path = DB_PATH, top_files: int = 4):
    """
    Stage 1: Rank files based on the average BM25 score of their top-K matching chunks.
    Prevents massive multi-topic pages from beating focused pages just because they have more chunks.
    """
    fts_query = _build_fts_query(query)

    if not fts_query:
        return []

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            filename,
            page_title,
            bm25(sections_fts, 0.0, 5.0, 2.5, 1.0) AS chunk_score
        FROM sections_fts
        WHERE sections_fts MATCH ?
        ORDER BY chunk_score ASC
        """,
        (fts_query,)
    )

    rows = cur.fetchall()
    conn.close()

    file_scores = {}

    for filename, page_title, chunk_score in rows:
        if filename not in file_scores:
            file_scores[filename] = {
                "filename": filename,
                "page_title": page_title,
                "matched_chunks": 0,
                "scores": [],
            }

        file_scores[filename]["matched_chunks"] += 1
        file_scores[filename]["scores"].append(chunk_score)

    ranked = []
    for info in file_scores.values():
        # BM25 is negative; smaller (more negative) = stronger match
        info["scores"].sort()
        top_slice = info["scores"][:TOP_CHUNKS_FOR_FILE_SCORE]
        avg_top_score = sum(top_slice) / len(top_slice)
        ranked.append({
            "filename": info["filename"],
            "page_title": info["page_title"],
            "matched_chunks": info["matched_chunks"],
            "best_score": info["scores"][0],
            "file_score": avg_top_score,
        })

    ranked.sort(key=lambda x: x["file_score"])
    return ranked[:top_files]


def search_sections_in_files(
    query: str,
    filenames: list[str],
    db_path: Path = DB_PATH,
    top_k_per_file: int = 3,
    max_total_chunks: int = 10,
):
    """
    Stage 2:
    Search selected files and retrieve the most relevant sections.
    """
    if not filenames:
        return []

    fts_query = _build_fts_query(query)

    if not fts_query:
        return []

    placeholders = ",".join("?" for _ in filenames)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    sql = f"""
        SELECT
            filename,
            page_title,
            section_title,
            body_text,
            bm25(sections_fts, 0.0, 3.0, 2.5, 1.5) AS bm25_score
        FROM sections_fts
        WHERE sections_fts MATCH ?
          AND filename IN ({placeholders})
        ORDER BY bm25_score ASC
    """

    params = [fts_query, *filenames]

    cur.execute(sql, params)
    rows = cur.fetchall()

    conn.close()

    grouped = {}

    for filename, page_title, section_title, body_text, score in rows:
        if filename not in grouped:
            grouped[filename] = []

        grouped[filename].append({
            "filename": filename,
            "page_title": page_title,
            "section_title": section_title,
            "body_text": body_text,
            "bm25_score": score,
        })

    selected_chunks = []

    for filename in filenames:
        file_chunks = grouped.get(filename, [])
        file_chunks.sort(key=lambda x: x["bm25_score"])
        selected_chunks.extend(file_chunks[:top_k_per_file])

    selected_chunks.sort(key=lambda x: x["bm25_score"])

    return selected_chunks[:max_total_chunks]


def search_kb_two_stage(
    query: str,
    db_path: Path = DB_PATH,
    top_files: int = 4,
    top_k_per_file: int = 3,
    max_total_chunks: int = 10,
):
    """
    Complete two-stage retrieval pipeline:
    1. Identify the most relevant files.
    2. Retrieve the most relevant sections from those files.
    """
    selected_files = search_top_files(
        query,
        db_path=db_path,
        top_files=top_files
    )

    filenames = [f["filename"] for f in selected_files]

    chunks = search_sections_in_files(
        query=query,
        filenames=filenames,
        db_path=db_path,
        top_k_per_file=top_k_per_file,
        max_total_chunks=max_total_chunks,
    )

    return selected_files, chunks


def format_two_stage_context(chunks, max_chars: int = 14000) -> str:
    """Group chunks cleanly by file/page so the LLM keeps each source distinct."""
    grouped = {}
    order = []

    for c in chunks:
        fname = c["filename"]
        if fname not in grouped:
            grouped[fname] = {
                "page_title": c["page_title"],
                "sections": [],
            }
            order.append(fname)
        grouped[fname]["sections"].append((c["section_title"], c["body_text"]))

    output_blocks = []
    used_chars = 0

    for fname in order:
        info = grouped[fname]
        header = f"=== DOCUMENT: {info['page_title']} (File: {fname}) ==="
        sec_texts = [f"[Section: {sec}]\n{body}" for sec, body in info["sections"]]
        doc_block = f"{header}\n\n" + "\n\n".join(sec_texts)

        if used_chars + len(doc_block) > max_chars:
            break

        output_blocks.append(doc_block)
        used_chars += len(doc_block)

    return "\n\n========================================\n\n".join(output_blocks)


if __name__ == "__main__":
    # Rebuild index once when run directly so duplicate collapsing & FTS5 stay fresh
    build_fts_index()

    print("\n" + "=" * 80)
    while True:
        # test_q = input("Enter your query (Enter nothing to exit): ").strip()
        # if not test_q:
        #     break

        test_query = [
                        "What is the annual BDS fee at Manipal College of Dental Sciences, Mangalore?",
                        "What is the NEET-UG eligibility criteria for BDS admission?",
                        "How many MDS seats are available at Government Dental College, Bangalore?",
                        "What is the duration of the MDS Orthodontics course at Saveetha Dental College, Chennai?",
                        "Does Jamia Millia Islamia offer a BDS course?",
                        "What is the annual hostel fee at AB Shetty Memorial Institute of Dental Sciences, Mangalore?",
                        "What is the NEET-MDS qualifying percentile for the General category?",
                        "How many BDS seats are available at Government Dental College, Mumbai?",
                        "Does SRM Kattankulathur Dental College offer MDS in Prosthodontics?",
                        "What is the annual tuition fee for BDS at Government Dental College, Thiruvananthapuram?",
                        "Compare the annual BDS tuition fees at Manipal College of Dental Sciences, Mangalore and Yenepoya Dental College, Mangalore.",
                        "What are the MDS Orthodontics fees and seat intake at Saveetha Dental College, Chennai?",
                        "Compare the BDS seat intake at Government Dental College, Bangalore and Government Dental College, Kozhikode.",
                        "What are the NEET-UG eligibility requirements and minimum qualifying marks for BDS admission?",
                        "What is the MDS Prosthodontics fee and duration at KLE Society's Institute of Dental Sciences, Bengaluru?",
                        "Does Nair Hospital Dental College, Mumbai offer MDS in Oral and Maxillofacial Surgery, and how many seats are available?",
                        "Compare the annual BDS fees at Dr. D. Y. Patil Dental College, Pune and Bharati Vidyapeeth Dental College, Pune.",
                        "What are the NEET-MDS eligibility requirements and internship completion requirements for admission?",
                        "What is the MDS Conservative Dentistry fee and seat intake at King George's Medical University, Lucknow?",
                        "Compare the number of MDS specializations offered by Government Dental College, Ahmedabad and Government Dental College, Jaipur",
                        "I completed my BDS from a recognized college, have completed my compulsory internship and qualified NEET-MDS. What are the eligibility requirements, documents and MDS seat intake at Postgraduate Institute of Dental Sciences, Rohtak?",
                        "I scored 450 in NEET-UG, belong to the General category and have Bihar domicile. What are the BDS eligibility requirements and previous-year closing cutoffs for Patna Dental College?",
                        "I scored 380 in NEET-UG, belong to the SC category and scored 55% in PCB in Class 12. What are the eligibility requirements for BDS at Government Dental College, Thrissur?",
                        "Compare Maulana Azad Institute of Dental Sciences, Delhi and Banaras Hindu University for BDS in terms of annual tuition fees, total seats and admission eligibility.",
                        "What are the eligibility requirements, compulsory internship deadline, annual fees and available seats for MDS Oral and Maxillofacial Surgery at King George's Medical University, Lucknow?",
                        "What is the complete annual cost of studying BDS at Kalinga Institute of Industrial Technology, Bhubaneswar, including tuition and hostel fees, and what are its seat intake and NEET-UG eligibility requirements?",
                        "I scored 520 in NEET-MDS, belong to the General category, completed BDS with 62% marks and have finished my internship. What are the eligibility requirements, MDS Prosthodontics seat intake and annual fees at Nair Hospital Dental College, Mumbai?",
                        "What are the BDS admission eligibility requirements, annual tuition fees, seat intake and any specific admission conditions at Aligarh Muslim University?",
                        "Compare SCB Dental College, Cuttack and Government Dental College, Ahmedabad in terms of BDS tuition fees, seat intake, NEET-UG eligibility and category-wise admission requirements.",
                        "Does JSS Dental College and Hospital, Mysuru offer MDS in Public Health Dentistry? If so, what are the annual fees, seat intake, course duration and eligibility requirements?"]    
        
        qcount = 1
        for query in test_query: 
            print("Query " + str(qcount) + ") " + query)
            files, chunks = search_kb_two_stage(
                query,
                top_files=4,
                top_k_per_file=3,
                max_total_chunks=10
            )

            print("STAGE 1 - SELECTED FILES:")
            for f in files:
                print(
                    f"  - {f['page_title']} ({f['filename']}) | "
                    # f"chunks={f['matched_chunks']} | avg_top={f['file_score']:.2f} | best={f['best_score']:.2f}"
                )

            # print("\nSTAGE 2 - RETRIEVED SECTIONS:")
            # for c in chunks:
            #     print(f"  - [{c['bm25_score']:.2f}] {c['filename']} -> {c['section_title']}")
            print("\n" + "-" * 80 + "\n")
            qcount += 1

        break
