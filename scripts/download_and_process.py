import requests
import fitz  # pymupdf
import json
import os
import argparse
import time
import xml.etree.ElementTree as ET
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

MODEL = "deepseek-ai/deepseek-v3.2"

# ── ArXiv ─────────────────────────────────────────────────────

def search_arxiv(query, max_results=5):
    try:
        r = requests.get("https://export.arxiv.org/api/query", params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }, timeout=30)
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            title = entry.find("atom:title", ns).text.strip()
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            entries.append((arxiv_id, title, pdf_url))
        return entries
    except Exception as e:
        print(f"ArXiv search error: {e}")
        return []

def download_pdf(pdf_url, save_path="/tmp/paper.pdf"):
    try:
        r = requests.get(pdf_url, stream=True, timeout=120,
                         headers={"User-Agent": "CooperTraining/1.0"})
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            print(f"Downloaded: {size_mb:.1f} MB")
            return True
        else:
            print(f"Download failed: HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"Download error: {e}")
        return False

# ── Text extraction ───────────────────────────────────────────

def extract_text_chunks(pdf_path, chunk_pages=8):
    try:
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
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

# ── DeepSeek call with thinking mode ─────────────────────────

def call_deepseek(prompt):
    """Call DeepSeek V3.2 with thinking enabled, return (thinking, response) tuple"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.95,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"thinking": True}},
            stream=True
        )
        thinking_text = ""
        response_text = ""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                thinking_text += reasoning
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        return thinking_text.strip(), response_text.strip()
    except Exception as e:
        print(f"  DeepSeek error: {e}")
        return "", ""

# ── SFT Q&A generation ────────────────────────────────────────

def chunk_to_qa(chunk, domain, paper_title):
    prompt = f"""You are generating training data for Cooper, a senior physical engineering reasoning AI.

This excerpt is from a recent research paper: "{paper_title}"

From this excerpt generate exactly 15 training examples.

STRICT RULES:
- Questions must be specific and technical — no vague questions
- Thinking must show full first-principles reasoning — identify physics, apply equations, show working, check units
- Response must read like a senior engineer — precise, numbered steps, values with units
- Include at least 3 cross-domain examples — where this concept appears in another engineering field
- Focus on novel research insights — what does this paper contribute beyond textbook knowledge

Return ONLY a valid JSON array. No markdown. No preamble. Schema:
[{{
  "prompt": "specific technical engineering question",
  "thinking": "1. State the physical problem\\n2. Identify relevant domains\\n3. List first principles and equations\\n4. Full step by step working with values\\n5. Sanity check",
  "response": "precise engineering answer with equations, numbers, units and domain references",
  "domain": "{domain}",
  "cross_domain": true
}}]

Paper excerpt:
{chunk[:7000]}"""

    thinking, response = call_deepseek(prompt)
    if not response:
        return []
    try:
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        records = json.loads(text)
        # Inject DeepSeek's actual thinking into the training records
        for r in records:
            if not r.get("thinking"):
                r["thinking"] = thinking[:2000] if thinking else ""
        return records
    except Exception as e:
        print(f"  Q&A parse error: {e}")
        return []

# ── DPO preference generation ─────────────────────────────────

def chunk_to_dpo(chunk, domain, paper_title):
    prompt = f"""You are generating DPO preference training data for Cooper, a senior engineering reasoning AI.

This excerpt is from a recent research paper: "{paper_title}"

From this excerpt generate exactly 8 preference pairs.

- chosen: senior engineer, 20 years experience — deep physics, cross-domain, first principles, specific numbers, references novel research
- rejected: junior engineer — correct but shallow, generic, missing physics depth

Return ONLY a valid JSON array. No markdown. No preamble. Schema:
[{{
  "prompt": "specific technical engineering question",
  "chosen": "senior engineer response — deep reasoning, first principles, cross-domain, specific",
  "rejected": "junior engineer response — correct but shallow and generic",
  "domain": "{domain}"
}}]

Paper excerpt:
{chunk[:7000]}"""

    _, response = call_deepseek(prompt)
    if not response:
        return []
    try:
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"  DPO parse error: {e}")
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
    print(f"Query: {book['title']}")
    print(f"Domain: {book['domain']}")
    print(f"{'='*50}\n")

    papers = search_arxiv(book["arxiv_query"], max_results=5)
    if not papers:
        print("ERROR: No papers found on ArXiv")
        return

    print(f"Found {len(papers)} papers")

    all_qa = []
    all_dpo = []

    for arxiv_id, title, pdf_url in papers:
        print(f"\nProcessing: {title}")
        print(f"ArXiv ID: {arxiv_id}")

        if not download_pdf(pdf_url):
            print("Skipping — download failed")
            continue

        chunks = extract_text_chunks("/tmp/paper.pdf")
        if not chunks:
            print("Skipping — no text extracted")
            os.remove("/tmp/paper.pdf")
            continue

        for i, chunk in enumerate(chunks):
            print(f"  Q&A chunk {i+1}/{len(chunks)}")
            qa = chunk_to_qa(chunk, book["domain"], title)
            all_qa.extend(qa)
            print(f"    → {len(qa)} records")

            print(f"  DPO chunk {i+1}/{len(chunks)}")
            dpo = chunk_to_dpo(chunk, book["domain"], title)
            all_dpo.extend(dpo)
            print(f"    → {len(dpo)} pairs")

            time.sleep(2)

        os.remove("/tmp/paper.pdf")
        time.sleep(3)

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
    print("Done.")

if __name__ == "__main__":
    main()
