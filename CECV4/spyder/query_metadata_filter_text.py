
import os
import json
from pathlib import Path

from groq import Groq
from dotenv import dotenv_values

# Import your existing metadata extractor
from query_metadata_extractor_test import extract_query_metadata


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_PATH = PROJECT_ROOT / ".env"

# Folder containing your reconstructed webpage text files
DOCUMENT_FOLDER = (
    PROJECT_ROOT
    / "spyder"
    / "CM_COMBINED_TEXT"
)

# Only these file types will be considered
ALLOWED_EXTENSIONS = {".txt"}

# Maximum number of documents Grok can select
MAX_SELECTED_FILES = 8


# ============================================================
# 2. LOAD GROQ API
# ============================================================

env_values = dotenv_values(ENV_PATH)

API_KEY = (
    env_values.get("GROQ_API_KEY")
    or os.getenv("GROQ_API_KEY")
)

if not API_KEY:
    raise ValueError(
        f"GROQ_API_KEY not found. Check: {ENV_PATH}"
    )

client = Groq(api_key=API_KEY)


# ============================================================
# 3. DOCUMENT INVENTORY
# ============================================================

def collect_document_names():

    """
    Collect all available parsed webpage filenames.

    Only filenames are collected.
    The actual webpage contents are NOT read.
    """

    if not DOCUMENT_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Document folder not found: {DOCUMENT_FOLDER}"
        )

    documents = sorted([
        file.name
        for file in DOCUMENT_FOLDER.iterdir()
        if file.is_file()
        and file.suffix.lower() in ALLOWED_EXTENSIONS
    ])

    if not documents:
        raise ValueError(
            "No parsed webpage files were found."
        )

    return documents


# ============================================================
# 4. DOCUMENT SELECTION SCHEMA
# ============================================================

SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_files": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "selected_files"
    ],
    "additionalProperties": False
}


# ============================================================
# 5. GROK DOCUMENT SELECTION PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a document selection engine for a dental
college information RAG chatbot.

Your task is to select the most relevant parsed webpage
files for answering a user's question.

You will receive:

1. The original user question.
2. Extracted query metadata.
3. A list of available webpage filenames.

YOUR TASK:

Examine the user's question and metadata.

Select the filenames that are most likely to contain
the information required to answer the question.

SELECTION RULES:

1. Only select filenames that exist in the supplied
   available filenames list.

2. Never invent, modify or guess filenames.

3. Prioritize documents matching the college,
   degree and user intent.

4. Consider the complete original question, not just
   the extracted metadata.

5. If the question concerns a particular college,
   prioritize documents belonging to that college.

6. If the question concerns a particular degree,
   prioritize documents associated with that degree.

7. If multiple documents are required to answer the
   question, select all relevant documents.

8. Do not select documents belonging to unrelated
   colleges simply because they discuss the same topic.

9. Do not select documents merely because their
   filenames contain similar keywords.

10. Select no more than 8 documents.

11. If no available document is relevant, return
    an empty selected_files array.

12. Do not assume that a document contains information
    merely because its filename suggests a topic.

IMPORTANT:
You are selecting documents, not answering the user.

Return ONLY JSON matching the supplied schema.
"""


# ============================================================
# 6. MODEL-BASED DOCUMENT SELECTION
# ============================================================

def select_documents(metadata):

    """
    Receives extracted query metadata and returns
    a validated list of relevant webpage filenames.
    """

    # Get the actual available filenames
    available_files = collect_document_names()

    # Prepare model input
    selection_input = {
        "original_query": metadata["original_query"],
        "query_metadata": metadata,
        "available_filenames": available_files
    }

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps(
                    selection_input,
                    ensure_ascii=False
                )
            }
        ],

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "document_selection",
                "strict": True,
                "schema": SELECTION_SCHEMA
            }
        },

        reasoning_effort="low",
        max_completion_tokens=1500
    )

    response_content = response.choices[0].message.content

    if not response_content:
        raise ValueError(
            "Grok returned an empty document selection."
        )

    result = json.loads(response_content)

    selected_files = result.get("selected_files", [])

    # ========================================================
    # 7. VALIDATE MODEL SELECTION
    # ========================================================

    # Only accept filenames that actually exist
    available_set = set(available_files)

    validated_files = []

    for filename in selected_files:

        if not isinstance(filename, str):
            continue

        if filename in available_set:
            if filename not in validated_files:
                validated_files.append(filename)

    # Enforce selection limit
    validated_files = validated_files[
        :MAX_SELECTED_FILES
    ]

    return validated_files


# ============================================================
# 8. COMPLETE QUERY ROUTING FUNCTION
# ============================================================

def route_query(user_query):

    """
    Complete pipeline:

    User query
        |
    Metadata extraction
        |
    Filename collection
        |
    Grok document selection
        |
    Filename validation
        |
    Return selected files
    """

    # STEP 1: Extract metadata
    metadata = extract_query_metadata(user_query)

    # STEP 2: Select relevant documents
    selected_files = select_documents(metadata)

    # STEP 3: Return results
    return {
        "metadata": metadata,
        "selected_files": selected_files
    }


# ============================================================
# 9. STANDALONE TERMINAL INTERFACE
# ============================================================

def run_terminal_router():

    print("\n" + "=" * 65)
    print("          CEC DOCUMENT ROUTER")
    print("=" * 65)

    print("\nModel:", MODEL_NAME)
    print("Document folder:", DOCUMENT_FOLDER)

    try:
        files = collect_document_names()

        print("Available webpages:", len(files))

    except Exception as e:
        print("Document inventory error:", e)
        return

    print("\nEnter a question to test document selection.")
    print("Type 'exit' or 'quit' to stop.")

    print("-" * 65)

    while True:

        try:

            query = input("\nUSER QUERY: ").strip()

            if query.lower() in ["exit", "quit"]:
                print("\nDocument router terminated.")
                break

            if not query:
                print("Please enter a valid question.")
                continue

            print("\n[1/3] Extracting query metadata...")

            result = route_query(query)

            print("\n[2/3] Extracted metadata:")

            print(json.dumps(
                result["metadata"],
                indent=4,
                ensure_ascii=False
            ))

            print("\n[3/3] Selected webpages:")

            selected = result["selected_files"]

            if selected:

                for index, filename in enumerate(
                    selected, start=1
                ):
                    print(f"{index}. {filename}")

                print(
                    f"\nTotal selected: {len(selected)}"
                )

            else:

                print(
                    "No relevant webpages were identified."
                )

            print("\n" + "-" * 65)

        except KeyboardInterrupt:

            print("\n\nDocument router terminated.")
            break

        except Exception as e:

            print(f"\nRouting failed: {e}")


# ============================================================
# 10. MODULE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_terminal_router()
