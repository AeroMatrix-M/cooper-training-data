import requests
import fitz
import json
import os
import argparse
import time
import xml.etree.ElementTree as ET
from openai import OpenAI

# SambaNova primary
sambanova_client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=os.environ["SAMBANOVA_API_KEY"]
)

# NVIDIA NIM fallback
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

SAMBANOVA_MODEL = "Meta-Llama-3.3-70B-Instruct"
NVIDIA_MODEL = "deepseek-ai/deepseek-v3.2"

# ── Source fetchers ───────────────────────────────────────────

def fetch_arxiv_direct(arxiv_id):
    try:
        r = requests.get("https://export.arxiv.org/api/query", params={
            "id_list": arxiv_id, "max_results": 1
        }, timeout=30)
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return []
        title = entry.find("atom:title", ns).text.strip()
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"ArXiv direct: {title}")
        return [(title, pdf_url)]
    except Exception as e:
        print(f"ArXiv direct error: {e}")
        return []

def fetch_arxiv_search(query, max_results=5):
    try:
        r = requests.get("https://export.arxiv.org/api/query", params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance"
        }, timeout=30)
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            title = entry.find("atom:title", ns).text.strip()
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            results.append((title, pdf_url))
        print(f"ArXiv search fallback: {len(results)} papers")
        return results
    except Exception as e:
        print(f"ArXiv search error: {e}")
        return []

def fetch_semantic_scholar(query, min_citations=100):
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": 20,
                "fields": "title,citationCount,openAccessPdf,externalIds,year"
            },
            headers={"User-Agent": "CooperTraining/1.0"},
            timeout=30
        )
        papers = r.json().get("data", [])
        qualified = []
        for p in papers:
            citations = p.get("citationCount", 0)
            pdf_info = p.get("openAccessPdf")
            ext_ids = p.get("externalIds", {})
            arxiv_id = ext_ids.get("ArXiv")

            # Use open access PDF or ArXiv ID fallback
            pdf_url = None
            if pdf_info and pdf_info.get("url"):
                pdf_url = pdf_info["url"]
            elif arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            if citations >= min_citations and pdf_url:
                qualified.append({
                    "title": p["title"],
                    "citations": citations,
                    "pdf_url": pdf_url,
                    "year": p.get("year", "?")
                })

        qualified.sort(key=lambda x: x["citations"], reverse=True)
        top = qualified[:5]
        print(f"Semantic Scholar: {len(papers)} found, {len(top)} qualified (>{min_citations} citations)")
        for p in top:
            print(f"  [{p['citations']} citations, {p['year']}] {p['title'][:70]}")

        # ArXiv fallback if nothing qualified
        if not top:
            print("No qualified papers found — falling back to ArXiv search")
            return fetch_arxiv_search(query)

        return [(p["title"], p["pdf_url"]) for p in top]

    except Exception as e:
        print(f"Semantic Scholar error: {e} — falling back to ArXiv")
        return fetch_arxiv_search(query)

def fetch_nasa_ntrs(query):
    try:
        r = requests.get(
            "https://ntrs.nasa.gov/api/citations/search",
            params={"keyword": query, "rows": 5, "start": 0},
            timeout=30
        )
        results = r.json().get("results", [])
        papers = []
        for item in results:
            doc_id = item.get("id")
            title = item.get("title", "NASA Report")
            pdf_url = f"https://ntrs.nasa.gov/api/citations/{doc_id}/downloads/legacy"
            papers.append((title, pdf_url))
        print(f"NASA NTRS: {len(papers)} reports found")
        for t, _ in papers:
            print(f"  {t[:70]}")

        # ArXiv fallback if nothing
        if not papers:
            print("No NASA reports found — falling back to ArXiv search")
            return fetch_arxiv_search(query)

        return papers
    except Exception as e:
        print(f"NASA NTRS error: {e} — falling back to ArXiv")
        return fetch_arxiv_search(query)

# ── PDF download and extraction ───────────────────────────────

def download_pdf(pdf_url, save_path="/tmp/paper.pdf"):
    try:
        r = requests.get(pdf_url, stream=True, timeout=120,
                         headers={"User-Agent": "CooperTraining/1.0"})
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            if size_mb < 0.05:
                print(f"File too small ({size_mb:.2f}MB) — skipping")
                return False
            print(f"Downloaded: {size_mb:.1f} MB")
            return True
        print(f"Download failed: HTTP {r.status_code}")
        return False
    except Exception as e:
        print(f"Download error: {e}")
        return False

def extract_text_chunks(pdf_path, chunk_pages=8):
    try:
        doc = fitz.open(pdf_path)
        print(f"Pages: {len(doc)}")
        chunks = []
        for i in range(0, len(doc), chunk_pages):
            text = ""
            for p in range(i, min(i + chunk_pages, len(doc))):
                text += doc[p].get_text()
            if len(text.strip()) > 300:
                chunks.append(text)
        print(f"Chunks: {len(chunks)}")
        return chunks
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

# ── LLM call SambaNova primary NVIDIA fallback ────────────────

def call_llm(prompt):
    try:
        response = sambanova_client.chat.completions.create(
            model=SAMBANOVA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  SambaNova error: {e} — trying NVIDIA")

    try:
        completion = nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.95,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"thinking": True}},
            stream=True
        )
        text = ""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            if chunk.choices[0].delta.content is not None:
                text += chunk.choices[0].delta.content
        return text.strip()
    except Exception as e:
        print(f"  NVIDIA error: {e}")
        return ""

# ── Data generation ───────────────────────────────────────────

def chunk_to_qa(chunk, domain, title, problem):
    prompt = f"""You are generating training data for Cooper, an AI co-engineer for physical engineering.

Cooper's purpose: when given an engineering problem, suggest creative technically grounded solutions by combining deep domain knowledge with cross-domain transfer — like a 20-year senior engineer who has worked across aerospace, robotics, marine, civil, medical and materials.

Paper: "{title}"
Core engineering problem this addresses: "{problem}"
Domain: {domain}

Generate exactly 20 training examples from this excerpt.

STRICT RULES:
- Frame questions as real engineering problem statements Cooper must solve
- Thinking must: (1) identify the physics, (2) check what other domains have solved this, (3) apply first principles with equations and numbers, (4) propose specific creative solutions
- Responses must propose concrete solutions — not just explain theory
- At least 6 of 20 must involve cross-domain insight (e.g. applying a biology solution to aerospace, or a naval solution to robotics)
- Include specific numbers, equations, and design parameters wherever possible
- Avoid generic textbook answers — Cooper suggests ideas, not recites facts

Return ONLY valid JSON array. No markdown. No preamble:
[{{
  "prompt": "Specific engineering problem statement to solve",
  "thinking": "1. Physics of the problem\\n2. What domains have solved analogous problems\\n3. First principles and equations\\n4. Proposed solution ideas with numbers\\n5. Best recommendation with justification",
  "response": "Senior engineer co-pilot response: specific creative solution proposals with physics backing, numbers, cross-domain references where relevant",
  "domain": "{domain}",
  "cross_domain": true or false,
  "problem_type": "design|analysis|optimisation|troubleshooting|concept"
}}]

Excerpt:
{chunk[:7000]}"""

    response = call_llm(prompt)
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
        print(f"  Q&A parse error: {e}")
        return []

def chunk_to_dpo(chunk, domain, title, problem):
    prompt = f"""You are generating DPO preference data for Cooper, an AI co-engineer.

Paper: "{title}"
Engineering problem: "{problem}"
Domain: {domain}

Generate exactly 10 preference pairs.

chosen — Cooper at its best:
- Identifies the core physics of the problem
- Draws on analogies from other engineering domains (aerospace to robotics, biology to structures etc)
- Proposes specific creative solution ideas with numbers and design parameters
- Acts like a brilliant engineering colleague suggesting ideas

rejected — mediocre response:
- Correct but shallow and generic
- Just explains theory without proposing solutions
- No cross-domain thinking
- Could be any textbook answer

Return ONLY valid JSON array. No markdown. No preamble:
[{{
  "prompt": "Engineering problem statement",
  "chosen": "Creative co-engineer response with specific solutions, cross-domain insight, numbers and physics",
  "rejected": "Generic shallow response, explains theory but proposes nothing creative or specific",
  "domain": "{domain}"
}}]

Excerpt:
{chunk[:7000]}"""

    response = call_llm(prompt)
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
    source = book.get("source", "semantic_scholar")
    problem = book.get("problem", "")

    print(f"\n{'='*55}")
    print(f"[{args.book_index}] {book['title']}")
    print(f"Source: {source} | Domain: {book['domain']}")
    print(f"Problem: {problem}")
    print(f"{'='*55}\n")

    # Fetch papers based on source type
    if source == "arxiv":
        papers = fetch_arxiv_direct(book["arxiv_id"])
    elif source == "semantic_scholar":
        papers = fetch_semantic_scholar(
            book["semantic_query"],
            book.get("min_citations", 100)
        )
    elif source == "nasa_ntrs":
        papers = fetch_nasa_ntrs(book["nasa_query"])
    else:
        print(f"Unknown source: {source}")
        return

    if not papers:
        print("ERROR: All sources exhausted — no papers fetched")
        return

    all_qa = []
    all_dpo = []

    for title, pdf_url in papers:
        print(f"\n--- {title[:70]}")
        if not download_pdf(pdf_url):
            print("Skipping — download failed")
            continue
        chunks = extract_text_chunks("/tmp/paper.pdf")
        if not chunks:
            print("Skipping — no text extracted")
            os.remove("/tmp/paper.pdf")
            continue
        for i, chunk in enumerate(chunks):
            print(f"  Q&A {i+1}/{len(chunks)}")
            qa = chunk_to_qa(chunk, book["domain"], title, problem)
            all_qa.extend(qa)
            print(f"    → {len(qa)} records")
            print(f"  DPO {i+1}/{len(chunks)}")
            dpo = chunk_to_dpo(chunk, book["domain"], title, problem)
            all_dpo.extend(dpo)
            print(f"    → {len(dpo)} pairs")
            time.sleep(1)
        os.remove("/tmp/paper.pdf")
        time.sleep(2)

    os.makedirs("dataset/qa_jsonl", exist_ok=True)
    qa_path = f"dataset/qa_jsonl/{book['domain']}_{args.book_index}.jsonl"
    with open(qa_path, "w") as f:
        for record in all_qa:
            f.write(json.dumps(record) + "\n")

    os.makedirs("dataset/dpo_jsonl", exist_ok=True)
    dpo_path = f"dataset/dpo_jsonl/{book['domain']}_{args.book_index}.jsonl"
    with open(dpo_path, "w") as f:
        for record in all_dpo:
            f.write(json.dumps(record) + "\n")

    print(f"\nQ&A: {len(all_qa)} → {qa_path}")
    print(f"DPO: {len(all_dpo)} → {dpo_path}")
    print("Done.")

if __name__ == "__main__":
    main()
