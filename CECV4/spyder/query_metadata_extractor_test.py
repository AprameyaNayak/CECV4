
import os
import json
from pathlib import Path
from groq import Groq
from dotenv import dotenv_values


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load API key from project root .env
ENV_PATH = PROJECT_ROOT / ".env"

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
# 2. METADATA SCHEMA
# ============================================================

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "degree": {
            "type": ["string", "null"]
        },
        "college": {
            "type": ["string", "null"]
        },
        "location": {
            "type": ["string", "null"]
        },
        "user_intent": {
            "type": "string"
        },
        "original_query": {
            "type": "string"
        }
    },
    "required": [
        "degree",
        "college",
        "location",
        "user_intent",
        "original_query"
    ],
    "additionalProperties": False
}


# ============================================================
# 3. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a query metadata extraction engine for a dental
college information RAG chatbot.

Extract EXACTLY these five fields:

1. degree
2. college
3. location
4. user_intent
5. original_query

RULES:

DEGREE:
- Identify MDS or BDS if mentioned or clearly implied.
- Return null if unidentified.
- Never invent a degree.

COLLEGE:
- Extract the full college or institution name.
- Return the name in lowercase.
- Return null if unidentified.

LOCATION:
- Extract a geographical location explicitly
  mentioned by the user.
- Return null if no location is mentioned.
- Never infer a location from the college name.

USER_INTENT:
- Identify the primary purpose of the question.
- Use a concise descriptive label.
- Examples: Fees, Admission, Eligibility,
  Hostel, Infrastructure, Courses, Comparison,
  Placement, Cutoff, General Information.
- These examples are not an exhaustive list.

ORIGINAL_QUERY:
- Preserve the user's original query exactly.
- Do not rewrite, summarize or correct it.

IMPORTANT:
- Do not create additional metadata fields.
- Do not hallucinate missing information.
- Return null for unidentified degree, college or location.
- Return only valid JSON matching the supplied schema.
"""


# ============================================================
# 4. CORE EXTRACTION FUNCTION
# ============================================================

def extract_query_metadata(user_query):

    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("Query cannot be empty.")

    original_query = user_query

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": original_query
            }
        ],

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "query_metadata",
                "strict": True,
                "schema": METADATA_SCHEMA
            }
        },

        reasoning_effort="low",
        max_completion_tokens=1000
    )

    response_content = response.choices[0].message.content

    if not response_content:
        raise ValueError("Empty response from Groq API.")

    metadata = json.loads(response_content)

    # Ensure original query is preserved
    metadata["original_query"] = original_query

    # Normalize degree
    if metadata["degree"]:
        degree = metadata["degree"].strip().upper()

        if degree not in ["MDS", "BDS"]:
            metadata["degree"] = None
        else:
            metadata["degree"] = degree

    # Normalize college
    if metadata["college"]:
        metadata["college"] = metadata["college"].strip().lower()

    # Normalize location
    if metadata["location"]:
        metadata["location"] = metadata["location"].strip()

    # Normalize intent
    metadata["user_intent"] = metadata["user_intent"].strip()

    return metadata


# ============================================================
# 5. STANDALONE TERMINAL INTERFACE
# ============================================================

def run_terminal_extractor():

    print("\n" + "=" * 60)
    print("       CEC QUERY METADATA EXTRACTOR")
    print("=" * 60)

    print("\nModel:", MODEL_NAME)

    print("\nEnter a question to extract its metadata.")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:

        try:

            query = input("\nUSER QUERY: ").strip()

            if query.lower() in ["exit", "quit"]:
                print("\nExtractor terminated.")
                break

            if not query:
                print("Please enter a valid question.")
                continue

            print("\nExtracting metadata...")

            metadata = extract_query_metadata(query)

            print("\n" + "-" * 60)
            print("EXTRACTED METADATA")
            print("-" * 60)

            print(json.dumps(
                metadata,
                indent=4,
                ensure_ascii=False
            ))

            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nExtractor terminated.")
            break

        except Exception as e:
            print(f"\nExtraction failed: {e}")


# ============================================================
# 6. MODULE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_terminal_extractor()
