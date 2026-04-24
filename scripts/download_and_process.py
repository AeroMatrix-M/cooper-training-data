import requests
import fitz  # pymupdf
import json
import os
import argparse
import time
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ── Archive.org ───────────────────────────────────────────────

def search_archive(query):
    try:
        r = requests.get("https://archive.org/advancedsearch.php", params={
            "q": query,
            "fl[]": ["identifier", "title"],
            "mediatype": "texts",
            "sort[]": "downloads desc",
            "rows": 5,
            "output": "json"
        }, timeout=30)
        data = r.json()
        docs = data.get("response", {}).get("docs", [])
        return docs[0]["identifier"] if docs else None
    except Exception as e:
        print(f"Archive search error: {e}")
        return None

def download_pdf(identifier, save_path="/tmp/book.pdf"):
    meta = requests.get(f"https://archive.org/metadata/{identifier}").json()
    for file in meta.get("files", []):
        if file["name"].endswith(".pdf"):
            url = f"https://archive.org/download/{identifier}/{file['name']}"
            print(f"Downloading from: {url}")
            r = requests.get(url, stream=True, timeout=120)
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            print(f"Downloaded: {size_mb:.1f} MB")
            return True
    print("No PDF found")
    return False

# ── Text extraction ───────────────────────────────────────────

def extract_text_chunks(pdf_path, chunk_pages=15):
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    chunks = []
    for i in range(0, len(doc), chunk_pages):
        text = ""
        for p in range(i, min(i + chunk_pages, len(doc))):
            text += doc[p].get_text()
        if len(text.strip()) > 300:
            chunks.append(text)
    print(f"Usable chunks: {len(chunks)}")
    return chunks

# ── Gemini SFT Q&A generation ─────────────────────────────────

def chunk_to_qa(chunk, domain):
    prompt = f"""
You are generating training data for Cooper, a senior physical engineering reasoning AI.

From this textbook excerpt generate exactly 15 training examples.

STRICT RULES:
- Questions must be specific and technical — no vague questions
- Thinking must show full first-principles reasoning — identify physics, apply equations, show working, check units
- Response must read like a senior engineer — precise, numbered steps, values with units
- Include at least 3 cross-domain examples per batch — where this engineering concept appears in another field

Return ONLY a valid JSON array. No markdown. No preamble. Schema:
[{{
  "prompt": "specific technical engineering question",
  "thinking": "1. State the physical problem\\n2. Identify relevant domains that have solved this\\n3. List first principles and equations\\n4. Full step by step working with values\\n5. Sanity check the answer",
  "response": "precise engineering answer with equations, numbers, units and domain references",
  "domain": "{domain}",
  "cross_domain": true
}}]

Textbook excerpt:
{chunk[:7000]}
"""
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"  Gemini Q&A error: {e}")
        return []

# ── Gemini DPO preference generation ─────────────────────────

def chunk_to_dpo(chunk, domain):
    prompt = f"""
You are generating DPO preference training data for Cooper, a senior engineering reasoning AI.

From this textbook excerpt generate exactly 8 preference pairs.

Each pair needs:
- chosen: how a senior engineer with 20 years experience would answer — deep physics, cross-domain awareness, first principles, specific numbers
- rejected: how a junior engineer would answer — correct but shallow, generic, missing physics depth

Return ONLY a valid JSON array. No markdown. No preamble. Schema:
[{{
  "prompt": "specific technical engineering question",
  "chosen": "senior engineer response — deep reasoning, first principles, cross-domain, specific",
  "rejected": "junior engineer response — correct but shallow and generic",
  "domain": "{domain}"
}}]

Textbook excerpt:
{chunk[:7000]}
"""
    try:
        response = gemini.generate_content(prompt)
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"  Gemini DPO error: {e}")
        return []

# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-index", type=int, required=True)
    args = parser.parse_args()

    with open("scripts/book_list.json") as f:
        books = json.load(f)

    book = books[args.book_index]
    print(f"\n{'='*50}")
    print(f"Book: {book['title']}")
    print(f"Domain: {book['domain']}")
    print(f"{'='*50}\n")

    identifier = search_archive(book["archive_query"])
    if not identifier:
        print("ERROR: Not found on Archive.org")
        return

    print(f"Found identifier: {identifier}")

    if not download_pdf(identifier):
        print("ERROR: Could not download PDF")
        return

    chunks = extract_text_chunks("/tmp/book.pdf")
    if not chunks:
        print("ERROR: No text extracted — likely scanned image PDF")
        os.remove("/tmp/book.pdf")
        return

    # Generate SFT Q&A and DPO pairs
    all_qa = []
    all_dpo = []

    for i, chunk in enumerate(chunks):
        print(f"Generating Q&A: chunk {i+1}/{len(chunks)}")
        qa = chunk_to_qa(chunk, book["domain"])
        all_qa.extend(qa)

        print(f"Generating DPO: chunk {i+1}/{len(chunks)}")
        dpo = chunk_to_dpo(chunk, book["domain"])
        all_dpo.extend(dpo)

        time.sleep(3)  # Gemini rate limit

    # Save SFT JSONL
    os.makedirs("dataset/qa_jsonl", exist_ok=True)
    qa_path = f"dataset/qa_jsonl/{book['domain']}_{args.book_index}.jsonl"
    with open(qa_path, "w") as f:
        for record in all_qa:
            f.write(json.dumps(record) + "\n")

    # Save DPO JSONL
    os.makedirs("dataset/dpo_jsonl", exist_ok=True)
    dpo_path = f"dataset/dpo_jsonl/{book['domain']}_{args.book_index}.jsonl"
    with open(dpo_path, "w") as f:
        for record in all_dpo:
            f.write(json.dumps(record) + "\n")

    print(f"\nQ&A records: {len(all_qa)} → {qa_path}")
    print(f"DPO pairs: {len(all_dpo)} → {dpo_path}")

    os.remove("/tmp/book.pdf")
    print("PDF deleted. Done.")

if __name__ == "__main__":
    main()
