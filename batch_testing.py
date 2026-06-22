import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

try:
    from judge_utils import (
        run_output_judges,
        run_retrieval_judges,
        run_case_meta_judge,
        run_kg_comparison_judges,
    )
    JUDGES_AVAILABLE = True
    JUDGE_IMPORT_ERROR = None
except Exception as e:
    JUDGES_AVAILABLE = False
    JUDGE_IMPORT_ERROR = str(e)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

INPUT_DIR = PROJECT_ROOT / "data" / "patient_notes"
OUTPUT_DIR = PROJECT_ROOT / "data" / "batch_outputs"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_search"
JUDGE_DEBUG_DIR = PROJECT_ROOT / "data" / "judge_debug"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LOCAL_MODEL = "gemma3:27b"
DEFAULT_JUDGE_MODEL = "ollama/gemma3:27b"

LOCAL_NUM_CTX = 30000
LOCAL_TIMEOUT = 600

BASE_URLS = [
    os.getenv("BASE_URL1"),
    os.getenv("BASE_URL2"),
]
BASE_URLS = [url for url in BASE_URLS if url]
ENDPOINT = os.getenv("ENDPOINT")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("VECTOR_DIR:", VECTOR_DIR)
print("JUDGE_DEBUG_DIR:", JUDGE_DEBUG_DIR)
print("BASE_URLS:", BASE_URLS)
print("ENDPOINT:", ENDPOINT)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

TOKEN_PATTERN = re.compile(r"\b[\w\-æøåÆØÅ\.]+\b", re.UNICODE)

PROMPT_LABELS = {
    "kg_full": "With KG + full PSO note",
    "no_kg": "Without KG",
    "kg_weighted": "With KG weighted as main evidence",
}

DEFAULT_PROMPT_TYPES = ["kg_full", "no_kg", "kg_weighted"]

DEFAULT_KG_EXPANSION_MODE = "multi-hop"
DEFAULT_MULTI_HOP_DEPTH = 3
DEFAULT_MAX_MULTIHOP_CANDIDATES = 500
DEFAULT_RELATION_RERANK_TOP_K = 40
DEFAULT_PENALIZE_HOP_DISTANCE = True
DEFAULT_HOP_1_WEIGHT = 1.0
DEFAULT_HOP_2_WEIGHT = 0.7
DEFAULT_HOP_3_WEIGHT = 0.4

DEFAULT_DIAGNOSIS_RETRIEVAL_MODE = "both"
DIAGNOSIS_RETRIEVAL_MODES = ["multi-term", "one-string", "both"]
DEFAULT_MULTI_TERM_PER_TERM_TOP_K = 10
DEFAULT_MULTI_TERM_FINAL_TOP_K = 30

DEFAULT_WEIGHT_SYMPTOMS = 1.2
DEFAULT_WEIGHT_FINDINGS = 1.3
DEFAULT_WEIGHT_TESTS = 0.9
DEFAULT_WEIGHT_ANATOMY = 0.7
DEFAULT_WEIGHT_DIAGNOSES = 1.5
DEFAULT_WEIGHT_CONTEXT = 0.8
DEFAULT_WEIGHT_ALIASES = 1.0
DEFAULT_WEIGHT_FALLBACK_TERMS = 1.0


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def safe_json(obj):
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(data), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(str(text or ""))
    os.replace(tmp_path, path)


def item_id(file_path, row_number, pso_note):
    raw = f"{file_path.name}|{row_number}|{pso_note}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{file_path.stem}_row_{row_number:05d}_{digest}"


def load_vector_data():
    with open(VECTOR_DIR / "relationship_index.json", "r", encoding="utf-8") as f:
        relationship_index = json.load(f)

    with open(VECTOR_DIR / "diagnosis_index.json", "r", encoding="utf-8") as f:
        diagnosis_index = json.load(f)

    with open(VECTOR_DIR / "relationship_lexical.json", "r", encoding="utf-8") as f:
        relationship_lexical = json.load(f)

    with open(VECTOR_DIR / "diagnosis_lexical.json", "r", encoding="utf-8") as f:
        diagnosis_lexical = json.load(f)

    relationship_embeddings = np.load(VECTOR_DIR / "relationship_embeddings.npy")
    diagnosis_embeddings = np.load(VECTOR_DIR / "diagnosis_embeddings.npy")

    relationship_lexical["tokenized_docs"] = [
        doc.split() if doc else [] for doc in relationship_lexical["tokenized_docs"]
    ]
    diagnosis_lexical["tokenized_docs"] = [
        doc.split() if doc else [] for doc in diagnosis_lexical["tokenized_docs"]
    ]

    relationship_id_to_index = {
        int(item["relationship_id"]): idx
        for idx, item in enumerate(relationship_index)
        if item.get("relationship_id") is not None
    }

    return {
        "relationship_index": relationship_index,
        "relationship_embeddings": relationship_embeddings,
        "relationship_lexical": relationship_lexical,
        "relationship_id_to_index": relationship_id_to_index,
        "diagnosis_index": diagnosis_index,
        "diagnosis_embeddings": diagnosis_embeddings,
        "diagnosis_lexical": diagnosis_lexical,
    }


def generate_local_response(prompt, model_name, base_url=None):
    if not BASE_URLS and not base_url:
        raise ValueError("Missing BASE_URL1/BASE_URL2/BASE_URL3 and/or ENDPOINT.")

    if not ENDPOINT:
        raise ValueError("Missing ENDPOINT.")

    if not model_name:
        raise ValueError("Local model name is missing.")

    selected_base_url = base_url or BASE_URLS[0]

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": LOCAL_NUM_CTX},
    }

    url = f"{selected_base_url}{ENDPOINT}"

    response = requests.post(
        url,
        json=payload,
        timeout=LOCAL_TIMEOUT,
    )

    if not response.ok:
        raise ValueError(
            f"{response.status_code} error from {url}: {response.text}"
        )

    data = response.json()

    if isinstance(data, dict):
        if "choices" in data:
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                if isinstance(message, dict) and message.get("content"):
                    return str(message["content"]).strip()
                if choices[0].get("text"):
                    return str(choices[0]["text"]).strip()

        if "message" in data:
            message = data.get("message") or {}
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"]).strip()

        if "response" in data and data.get("response"):
            return str(data["response"]).strip()

    raise ValueError(f"Could not parse local LLM response from {url}: {data}")


def generate_llm_response(prompt, model_name, base_url=None):
    return generate_local_response(
        prompt=prompt,
        model_name=model_name,
        base_url=base_url,
    )


def parse_json_response(text):
    text = str(text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start:end + 1])


def tokenize(text):
    if not text:
        return []
    return [t.lower() for t in TOKEN_PATTERN.findall(str(text))]


def normalize_scores(scores):
    arr = np.array(scores, dtype=float)
    if len(arr) == 0:
        return arr
    min_v = arr.min()
    max_v = arr.max()
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def lexical_score(query_text, doc_tokens, idf):
    query_tokens = tokenize(query_text)
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_counter = Counter(doc_tokens)
    score = 0.0
    for token in query_tokens:
        if token in doc_counter:
            score += idf.get(token, 1.0)
    return float(score)


def compute_dense_scores(query, embeddings, model):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return (query_embedding @ embeddings.T)[0]


def hybrid_rank(query, index_items, embeddings, lexical_data, model, alpha=0.7, beta=0.3, top_k=None):
    dense_scores = compute_dense_scores(query, embeddings, model)
    lexical_scores = np.array([
        lexical_score(query, lexical_data["tokenized_docs"][i], lexical_data["idf"])
        for i in range(len(index_items))
    ])

    dense_norm = normalize_scores(dense_scores)
    lexical_norm = normalize_scores(lexical_scores)
    hybrid_scores = alpha * dense_norm + beta * lexical_norm
    ranked_idx = np.argsort(hybrid_scores)[::-1]

    if top_k is not None:
        ranked_idx = ranked_idx[:top_k]

    results = []
    for rank_position, idx in enumerate(ranked_idx):
        item = index_items[idx]
        next_score = float(hybrid_scores[ranked_idx[rank_position + 1]]) if rank_position + 1 < len(ranked_idx) else None

        row = dict(item)
        row["dense_score"] = float(dense_scores[idx])
        row["lexical_score"] = float(lexical_scores[idx])
        row["score"] = float(hybrid_scores[idx])
        row["next_score"] = next_score
        row["score_gap_to_next"] = float(hybrid_scores[idx] - next_score) if next_score is not None else None
        results.append(row)

    return results


def search_diagnoses(query, diagnosis_index, diagnosis_embeddings, diagnosis_lexical, model, top_k=30):
    return hybrid_rank(
        query=query,
        index_items=diagnosis_index,
        embeddings=diagnosis_embeddings,
        lexical_data=diagnosis_lexical,
        model=model,
        alpha=0.7,
        beta=0.3,
        top_k=top_k,
    )


def split_retrieval_terms(query):
    terms = [t.strip() for t in str(query).split("|") if t.strip()]
    return terms if terms else [str(query).strip()]


def build_query_enhancement_prompt(user_text):
    return f"""
You are converting a clinical patient note into retrieval terms for first-stage search over diagnosis-like nodes in a Danish medical knowledge graph.

Return valid JSON only with the following keys:
- symptoms: list[str]
- findings: list[str]
- test_or_measurement_terms: list[str]
- anatomy_terms: list[str]
- possible_diagnoses_if_supported: list[str]
- clinically_relevant_risk_or_context_terms: list[str]
- lexical_variants_and_aliases: list[str]
- diagnosis_retrieval_query: str

Goal:
Create a broad but clinically focused retrieval representation that improves recall when searching diagnosis-like nodes.

General output rules:
- Use short atomic keyword-like clinical terms.
- Do not write sentences or summaries.
- Prefer Danish terms.
- Do not use English translations.
- Do not use abbreviations.
- Expand abbreviations into full Danish wording.
- Prefer understandable Danish clinical wording when possible.
- Keep only clinically useful retrieval terms.

Very important extraction rules:
- Extract only terms that are explicitly present in the note, or very tight normalized reformulations of explicitly present content.
- Do not invent extra symptoms, diagnoses, findings, or disease categories.
- Do not generalize a specific finding into a broader disease label unless that diagnosis is clearly supported by the note.
- Do not add related concepts just because they are medically associated.
- Do not add broader umbrella terms when a more specific extracted term already exists.
- Do not add vague or generic terms that are weaker than the original wording.

Priority of content to include:
1. symptoms
2. abnormal clinical findings
3. diagnostically useful tests, measurements, and test results written in full wording
4. anatomy or body location
5. possible diagnoses only if clearly supported by the note
6. clinically important risk or context only if it directly helps diagnosis retrieval

Strict exclusion rules:
- Exclude medications unless diagnostically important for the current problem.
- Exclude incidental comorbidities unless relevant to the current diagnostic problem.
- Exclude past history unrelated to the current presentation.
- Exclude background history that is not useful for diagnosis retrieval.
- Exclude long narrative phrases.
- Exclude administrative or social details unless diagnostically important.
- Exclude explicitly normal findings unless highly important for narrowing diagnosis.
- Exclude negated findings unless highly important for narrowing diagnosis.
- Exclude weak contextual terms that are unlikely to help retrieve diagnosis-like nodes.
- Exclude broad disease classes unless clearly supported.
- Exclude terms that are only loosely related to the note.

Normalization rules:
- Prefer concise canonical forms.
- Replace abbreviations with full Danish wording.
- Prefer readable Danish wording over unnecessarily technical wording.
- Convert long phrases to the shortest clinically meaningful retrieval term.
- Only include lexical variants that are close Danish variants of the same concept.
- Do not create new concepts during normalization.

diagnosis_retrieval_query rules:
- Must be a compact separator-based list of high-value retrieval terms.
- Must contain only clinically useful Danish retrieval terms.
- Must not be prose.
- Must not contain explanatory text.
- Must not contain abbreviations.
- Must reflect the highest-value retrieval concepts from the note.
- Must avoid incidental history and weak context.

Clinical note:
{user_text}
""".strip()


def extract_structured_query(user_text, model_name, max_retries=3, base_delay=2, base_url=None):
    prompt = build_query_enhancement_prompt(user_text)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = generate_llm_response(
                prompt=prompt,
                model_name=model_name,
                base_url=base_url,
            )
            data = parse_json_response(response)

            expected_keys = [
                "symptoms",
                "findings",
                "test_or_measurement_terms",
                "anatomy_terms",
                "possible_diagnoses_if_supported",
                "clinically_relevant_risk_or_context_terms",
                "lexical_variants_and_aliases",
            ]

            clean = {}
            for key in expected_keys:
                value = data.get(key, [])
                clean[key] = [str(x).strip() for x in value if str(x).strip()]

            diagnosis_query = str(data.get("diagnosis_retrieval_query", "")).strip()
            if not diagnosis_query:
                raise ValueError("Empty diagnosis_retrieval_query.")

            clean["diagnosis_retrieval_query"] = diagnosis_query
            return clean

        except Exception as e:
            last_error = e
            error_text = str(e).lower()

            retryable = any(x in error_text for x in [
                "503", "unavailable", "rate limit", "timeout",
                "overloaded", "connection aborted", "temporarily unavailable",
            ])

            if not retryable or attempt == max_retries - 1:
                break

            time.sleep(base_delay * (attempt + 1))

    raise last_error


def fallback_query_enhancement(user_text):
    tokens = tokenize(user_text)
    unique_tokens = []
    seen = set()

    for token in tokens:
        token = token.strip().lower()
        if len(token) <= 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)

    diagnosis_query = " ".join(unique_tokens[:80])

    return {
        "symptoms": [],
        "findings": [],
        "test_or_measurement_terms": [],
        "anatomy_terms": [],
        "possible_diagnoses_if_supported": [],
        "clinically_relevant_risk_or_context_terms": [],
        "lexical_variants_and_aliases": [],
        "diagnosis_retrieval_query": diagnosis_query if diagnosis_query else user_text.strip(),
    }


def build_diagnosis_retrieval_query_from_structured(structured):
    ordered_terms = (
        structured.get("symptoms", [])
        + structured.get("findings", [])
        + structured.get("test_or_measurement_terms", [])
        + structured.get("possible_diagnoses_if_supported", [])
        + structured.get("lexical_variants_and_aliases", [])
        + structured.get("anatomy_terms", [])
        + structured.get("clinically_relevant_risk_or_context_terms", [])
    )

    clean_terms = []
    seen = set()

    for term in ordered_terms:
        term = str(term).strip()
        if not term:
            continue
        norm = term.lower()
        if norm in seen:
            continue
        seen.add(norm)
        clean_terms.append(term)

    fallback_query = str(structured.get("diagnosis_retrieval_query", "")).strip()
    return " | ".join(clean_terms) if clean_terms else fallback_query


def build_weighted_terms_from_structured_query(structured, fallback_query=None, weights=None):
    weights = weights or {}

    ordered_categories = [
        "symptoms",
        "findings",
        "test_or_measurement_terms",
        "possible_diagnoses_if_supported",
        "lexical_variants_and_aliases",
        "anatomy_terms",
        "clinically_relevant_risk_or_context_terms",
    ]

    weighted_terms = []
    seen = set()

    for category in ordered_categories:
        for term in structured.get(category, []) or []:
            term = str(term).strip()
            if not term:
                continue
            norm = term.lower()
            if norm in seen:
                continue
            seen.add(norm)
            weighted_terms.append({
                "term": term,
                "weight": float(weights.get(category, 1.0)),
                "category": category,
            })

    if not weighted_terms:
        weighted_terms = [
            {
                "term": term,
                "weight": float(weights.get("fallback_terms", 1.0)),
                "category": "fallback_terms",
            }
            for term in split_retrieval_terms(fallback_query or structured.get("diagnosis_retrieval_query", ""))
        ]

    return weighted_terms


def build_weighted_terms_from_query_string(query, default_weight=1.0):
    return [
        {"term": term, "weight": float(default_weight), "category": "query_string"}
        for term in split_retrieval_terms(query)
    ]


def multi_term_search_diagnoses(
    query,
    diagnosis_index,
    diagnosis_embeddings,
    diagnosis_lexical,
    model,
    structured_query=None,
    per_term_top_k=10,
    final_top_k=30,
    weights=None,
):
    if structured_query:
        weighted_terms = build_weighted_terms_from_structured_query(
            structured=structured_query,
            fallback_query=query,
            weights=weights,
        )
    else:
        weighted_terms = build_weighted_terms_from_query_string(
            query=query,
            default_weight=weights.get("fallback_terms", 1.0) if weights else 1.0,
        )

    node_scores = {}

    for term_info in weighted_terms:
        term = term_info["term"]
        term_weight = float(term_info.get("weight", 1.0))
        category = term_info.get("category", "")

        term_results = search_diagnoses(
            query=term,
            diagnosis_index=diagnosis_index,
            diagnosis_embeddings=diagnosis_embeddings,
            diagnosis_lexical=diagnosis_lexical,
            model=model,
            top_k=per_term_top_k,
        )

        for rank, item in enumerate(term_results, start=1):
            node_id = item.get("node_id")
            if node_id is None:
                continue

            contribution = float(item.get("score", 0.0)) * (1 / rank) * term_weight

            if node_id not in node_scores:
                row = dict(item)
                row["score"] = 0.0
                row["matched_terms"] = []
                row["matched_categories"] = []
                row["term_contributions"] = []
                node_scores[node_id] = row

            node_scores[node_id]["score"] += contribution
            node_scores[node_id]["matched_terms"].append(term)
            node_scores[node_id]["matched_categories"].append(category)
            node_scores[node_id]["term_contributions"].append({
                "term": term,
                "category": category,
                "term_weight": term_weight,
                "term_score": float(item.get("score", 0.0)),
                "rank": rank,
                "rank_weight": 1 / rank,
                "contribution": contribution,
            })

    results = list(node_scores.values())

    for row in results:
        row["matched_term_count"] = len(set(row.get("matched_terms", [])))
        row["matched_category_count"] = len(set(row.get("matched_categories", [])))

    results.sort(
        key=lambda x: (
            float(x.get("score", 0.0)),
            int(x.get("matched_term_count", 0)),
            int(x.get("matched_category_count", 0)),
        ),
        reverse=True,
    )

    return results[:final_top_k]


def retrieve_diagnosis_candidates(
    query,
    diagnosis_index,
    diagnosis_embeddings,
    diagnosis_lexical,
    model,
    retrieval_mode="both",
    structured_query=None,
    top_k=30,
    per_term_top_k=10,
    weights=None,
):
    if retrieval_mode == "one-string":
        return {
            "one-string": search_diagnoses(
                query, diagnosis_index, diagnosis_embeddings, diagnosis_lexical, model, top_k
            )
        }

    if retrieval_mode == "multi-term":
        return {
            "multi-term": multi_term_search_diagnoses(
                query, diagnosis_index, diagnosis_embeddings, diagnosis_lexical,
                model, structured_query, per_term_top_k, top_k, weights
            )
        }

    if retrieval_mode == "both":
        return {
            "one-string": search_diagnoses(
                query, diagnosis_index, diagnosis_embeddings, diagnosis_lexical, model, top_k
            ),
            "multi-term": multi_term_search_diagnoses(
                query, diagnosis_index, diagnosis_embeddings, diagnosis_lexical,
                model, structured_query, per_term_top_k, top_k, weights
            ),
        }

    raise ValueError(f"Unknown diagnosis retrieval mode: {retrieval_mode}")


def build_diagnosis_rerank_prompt(patient_note, candidates):
    candidate_lines = []

    for i, c in enumerate(candidates, start=1):
        desc = (c.get("description", "") or "")[:300]
        candidate_lines.append(
            f"{i}. candidate_id={c['candidate_id']} | "
            f"name={c.get('name', '')} | "
            f"label={c.get('label', '')} | "
            f"description={desc}"
        )

    candidate_block = "\n".join(candidate_lines)

    return f"""
You are evaluating the clinical relevance of diagnosis or condition nodes for medical knowledge graph retrieval.

Task:
Given the patient note and candidate diagnosis or condition nodes, evaluate EVERY candidate independently.
Assign each candidate a clinical relevance score from 0.0 to 1.0.
Also decide which candidates should be used for downstream relation expansion.

Return valid JSON only with this schema:
{{
  "scored_candidates": [
    {{
      "candidate_id": "string",
      "relevance_score": 0.0,
      "include_for_expansion": true,
      "reason": "short reason"
    }}
  ]
}}

Rules:
- You MUST return exactly one entry for EVERY provided candidate.
- The number of items in "scored_candidates" MUST equal the number of input candidates.
- Every candidate_id MUST appear exactly once.
- Do NOT omit any candidates.
- Do NOT add new candidates.
- Include candidates with partial relevance if they may help downstream KG expansion.
- Score each candidate independently.
- Use low scores for weak candidates instead of skipping them.
- Exclude only candidates that are clearly unrelated or clinically misleading.
- Do not guess unsupported facts.
- Keep reasons short.

Patient note:
{patient_note.strip()}

Candidate nodes:
{candidate_block}
""".strip()


def rerank_diagnosis_candidates(patient_note, candidates, model_name, base_url=None):
    enriched = []
    for idx, c in enumerate(candidates):
        row = dict(c)
        row["candidate_id"] = str(c.get("node_id", f"cand_{idx}"))
        enriched.append(row)

    prompt = build_diagnosis_rerank_prompt(patient_note, enriched)

    response = generate_llm_response(
        prompt=prompt,
        model_name=model_name,
        base_url=base_url,
    )

    data = parse_json_response(response)
    scored = data.get("scored_candidates", [])

    score_map = {
        str(item["candidate_id"]): item
        for item in scored
        if isinstance(item, dict) and "candidate_id" in item
    }

    reranked = []

    for c in enriched:
        cid = str(c["candidate_id"])
        row = dict(c)
        llm_data = score_map.get(cid) or {
            "relevance_score": row.get("score", 0.0),
            "include_for_expansion": False,
            "reason": "LLM did not score this item",
        }

        row["llm_relevance"] = float(llm_data.get("relevance_score", 0.0))
        row["include_for_expansion"] = bool(llm_data.get("include_for_expansion", False))
        row["payload_reason"] = str(llm_data.get("reason", "")).strip()
        reranked.append(row)

    reranked.sort(key=lambda x: x.get("llm_relevance", 0.0), reverse=True)

    return {
        "all_candidates": reranked,
        "selected_candidates": [x for x in reranked if float(x.get("llm_relevance", 0.0)) > 0],
        "prompt": prompt,
        "raw_response": response,
    }


def create_neo4j_driver():
    if not NEO4J_URI:
        raise ValueError("Missing NEO4J_URI.")

    if not NEO4J_USERNAME:
        raise ValueError("Missing NEO4J_USERNAME.")

    if not NEO4J_PASSWORD:
        raise ValueError("Missing NEO4J_PASSWORD.")

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


def collect_relation_candidates_from_neo4j(
    neo4j_driver,
    selected_node_ids,
    max_depth=3,
    max_candidates=500,
):
    selected_node_ids = [
        int(node_id)
        for node_id in selected_node_ids
        if node_id is not None
    ]

    if not selected_node_ids:
        return {}

    max_depth = max(1, int(max_depth))
    max_candidates = max(1, int(max_candidates))

    query = f"""
    MATCH (start:Entity)
    WHERE start.node_id IN $selected_node_ids

    MATCH path = (start)-[*1..{max_depth}]-(neighbor:Entity)
    WHERE all(rel IN relationships(path) WHERE rel.relationship_id IS NOT NULL)

    UNWIND relationships(path) AS rel

    WITH
        rel.relationship_id AS relationship_id,
        min(length(path)) AS hop_depth

    WHERE relationship_id IS NOT NULL

    RETURN
        relationship_id,
        hop_depth

    ORDER BY hop_depth ASC, relationship_id ASC
    LIMIT $max_candidates
    """

    relation_depths_by_relationship_id = {}

    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        rows = session.run(
            query,
            selected_node_ids=selected_node_ids,
            max_candidates=max_candidates,
        )

        for row in rows:
            relationship_id = row["relationship_id"]
            hop_depth = row["hop_depth"]

            if relationship_id is not None:
                relation_depths_by_relationship_id[int(relationship_id)] = int(hop_depth)

    return relation_depths_by_relationship_id


def build_relation_adjacency(relationship_index):
    adjacency = {}

    for i, rel in enumerate(relationship_index):
        source = rel.get("source_node_id")
        target = rel.get("target_node_id")

        if source is None or target is None:
            continue

        adjacency.setdefault(source, []).append((target, i))
        adjacency.setdefault(target, []).append((source, i))

    return adjacency


def collect_multihop_relation_indices(selected_node_ids, relationship_index, max_depth=3, max_candidates=500):
    adjacency = build_relation_adjacency(relationship_index)
    visited_nodes = set(selected_node_ids)
    frontier = set(selected_node_ids)
    relation_depths = {}

    for depth in range(1, max_depth + 1):
        next_frontier = set()

        for node_id in frontier:
            for neighbor_id, rel_idx in adjacency.get(node_id, []):
                if rel_idx not in relation_depths:
                    relation_depths[rel_idx] = depth

                    if len(relation_depths) >= max_candidates:
                        return relation_depths

                if neighbor_id not in visited_nodes:
                    visited_nodes.add(neighbor_id)
                    next_frontier.add(neighbor_id)

        frontier = next_frontier

        if not frontier:
            break

    return relation_depths


def collect_relation_candidates_from_selected_nodes(
    selected_node_ids,
    relationship_index,
    expansion_mode="multi-hop",
    multi_hop_depth=3,
    max_multihop_candidates=500,
):
    return collect_multihop_relation_indices(
        selected_node_ids=selected_node_ids,
        relationship_index=relationship_index,
        max_depth=multi_hop_depth,
        max_candidates=max_multihop_candidates,
    )


def add_hop_depth_to_candidate_items(candidate_items, candidate_indices, relation_depths):
    enriched = []

    for item, original_idx in zip(candidate_items, candidate_indices):
        row = dict(item)
        row["hop_depth"] = int(relation_depths.get(original_idx, 1))
        enriched.append(row)

    return enriched


def apply_hop_distance_penalty(results, enabled=True, hop_weights=None):
    hop_weights = hop_weights or {1: 1.0, 2: 0.7, 3: 0.4}
    adjusted = []

    for row in results:
        row = dict(row)
        hop_depth = int(row.get("hop_depth", 1))
        hop_weight = float(hop_weights.get(hop_depth, hop_weights.get(3, 0.4)))

        row["original_score"] = row.get("score", 0.0)
        row["hop_weight"] = hop_weight

        if enabled:
            row["score"] = float(row.get("score", 0.0)) * hop_weight

        adjusted.append(row)

    adjusted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return adjusted


def build_relation_rerank_prompt(patient_note, candidates):
    candidate_lines = []

    for i, r in enumerate(candidates, start=1):
        desc = (r.get("description", "") or "")[:300]
        candidate_lines.append(
            f"{i}. relationship_id={r['relationship_id']} | "
            f"source_name={r.get('source_name', '')} | "
            f"source_label={r.get('source_label', '')} | "
            f"type={r.get('type', '')} | "
            f"target_name={r.get('target_name', '')} | "
            f"target_label={r.get('target_label', '')} | "
            f"description={desc}"
        )

    candidate_block = "\n".join(candidate_lines)

    return f"""
You are evaluating the clinical relevance of knowledge graph relations for medical reasoning.

Task:
Given the patient note and candidate knowledge graph relations, evaluate EVERY relation independently.
Assign each relation a clinical relevance score from 0.0 to 1.0.
Also decide which relations should be included in the final knowledge graph payload.

Return valid JSON only with this schema:
{{
  "scored_relations": [
    {{
      "relationship_id": "string",
      "relevance_score": 0.0,
      "include_in_payload": true,
      "reason": "short reason"
    }}
  ]
}}

Rules:
- You MUST return exactly one entry for EVERY provided relation.
- The number of items in "scored_relations" MUST equal the number of input relations.
- Every relationship_id MUST appear exactly once.
- Do NOT omit any relations.
- Do NOT add new relations.
- Include relations with partial relevance if they contribute to understanding diagnosis, symptoms, or mechanisms.
- Score each relation independently.
- Use low scores for weak relations instead of skipping them.
- Exclude only relations that are clearly unrelated or clinically misleading.
- Prefer relations that connect symptoms, findings, diagnoses, or mechanisms relevant to the patient note.
- Do not guess unsupported facts.
- Keep reasons short.

Patient note:
{patient_note.strip()}

Candidate relations:
{candidate_block}
""".strip()


def rerank_relation_candidates(patient_note, candidates, model_name, base_url=None):
    enriched = []

    for idx, r in enumerate(candidates):
        row = dict(r)
        row["relationship_id"] = str(r.get("relationship_id", f"rel_{idx}"))
        enriched.append(row)

    prompt = build_relation_rerank_prompt(patient_note, enriched)

    response = generate_llm_response(
        prompt=prompt,
        model_name=model_name,
        base_url=base_url,
    )

    data = parse_json_response(response)
    scored = data.get("scored_relations", [])

    score_map = {
        str(item["relationship_id"]): item
        for item in scored
        if isinstance(item, dict) and "relationship_id" in item
    }

    reranked = []

    for r in enriched:
        rid = str(r["relationship_id"])
        row = dict(r)
        llm_data = score_map.get(rid) or {
            "relevance_score": row.get("score", 0.0),
            "include_in_payload": False,
            "reason": "LLM did not score this item",
        }

        row["llm_relevance"] = float(llm_data.get("relevance_score", 0.0))
        row["include_in_payload"] = bool(llm_data.get("include_in_payload", False))
        row["payload_reason"] = str(llm_data.get("reason", "")).strip()
        reranked.append(row)

    reranked.sort(key=lambda x: x.get("llm_relevance", 0.0), reverse=True)

    return {
        "all_candidates": reranked,
        "selected_candidates": [x for x in reranked if float(x.get("llm_relevance", 0.0)) > 0],
        "prompt": prompt,
        "raw_response": response,
    }


def build_relation_payload(query, relevant_relations):
    return {
        "user_query": query,
        "selected_relations": [
            {
                "relationship_id": rel.get("relationship_id"),
                "source_node_id": rel.get("source_node_id"),
                "source_name": rel.get("source_name", ""),
                "source_label": rel.get("source_label", ""),
                "source_description": rel.get("source_description", ""),
                "target_node_id": rel.get("target_node_id"),
                "target_name": rel.get("target_name", ""),
                "target_label": rel.get("target_label", ""),
                "target_description": rel.get("target_description", ""),
                "type": rel.get("type", ""),
                "description": rel.get("description", ""),
                "relevance_score": rel.get("llm_relevance"),
                "vector_score": rel.get("score"),
                "sources": rel.get("sources", []),
            }
            for rel in relevant_relations
        ],
    }

def format_diagnosis_candidates_for_label_match(diagnosis_result_sets, max_items_per_set=50):
    lines = []

    for search_name, candidates in (diagnosis_result_sets or {}).items():
        lines.append(f"{search_name}:")

        for idx, item in enumerate((candidates or [])[:max_items_per_set], start=1):
            node_id = item.get("node_id", "")
            name = item.get("name", "")
            normalized_name = item.get("normalized_name", "")
            label = item.get("label", "")
            label_group = item.get("label_group", "")
            score = item.get("score", None)

            score_text = f"{float(score):.4f}" if score is not None else "N/A"

            lines.append(
                f"{idx}. node_id={node_id} | "
                f"name={name} | "
                f"normalized_name={normalized_name} | "
                f"label={label} | "
                f"label_group={label_group} | "
                f"score={score_text}"
            )

        lines.append("")

    return "\n".join(lines).strip()

def format_kg_relations_for_prompt(payload):
    if not payload.get("selected_relations"):
        return "No relevant triplets were retrieved."

    lines = []

    for rel in payload["selected_relations"]:
        relevance_score = rel.get("relevance_score")
        vector_score = rel.get("vector_score")

        relevance_text = f"{relevance_score:.2f}" if relevance_score is not None else "N/A"
        vector_text = f"{vector_score:.2f}" if vector_score is not None else "N/A"

        line = (
            f"- {rel['source_name']} --{rel['type']}--> {rel['target_name']} "
            f"| relevance_score={relevance_text} "
            f"| vector_score={vector_text}"
        )

        if rel.get("description"):
            line += f" | {rel['description']}"

        lines.append(line)

    return "\n".join(lines)


def build_kg_interpretation_instruction():
    return """
Knowledge graph interpretation:
- relevance_score is the LLM-assessed clinical relevance of the triplet for this patient note.
- vector_score is the retrieval similarity score.
- Prefer triplets with higher relevance_score.
- Ignore irrelevant or contradicted triplets.
""".strip()


def build_no_kg_prompt(patient_note):
    return f"""
You are given a patient note.

Patient note:
{patient_note}

Task:
Return:
1. Most likely diagnosis
2. Brief reasoning
3. Important differentials
4. Suggested next clinical checks
""".strip()


def build_relation_payload_message(payload):
    kg_text = format_kg_relations_for_prompt(payload)

    return f"""
You are given a patient note and relevant knowledge graph triplets.

Patient note:
{payload["user_query"]}

Relevant knowledge graph triplets:
{kg_text}

{build_kg_interpretation_instruction()}

Task:
Return:
1. Most likely diagnosis
2. Brief reasoning
3. Important differentials
4. Suggested next clinical checks
""".strip()


def build_relation_payload_message_kg_weighted(payload):
    kg_text = format_kg_relations_for_prompt(payload)

    return f"""
You are given a patient note and relevant knowledge graph triplets.

Patient note:
{payload["user_query"]}

Relevant knowledge graph triplets:
{kg_text}

{build_kg_interpretation_instruction()}

Task:
Return:
1. Most likely diagnosis
2. Brief reasoning
3. Important differentials
4. Suggested next clinical checks

Important:
Use the knowledge graph triplets as the main diagnostic evidence.
The patient note provides clinical context.
Clearly mention when KG evidence strongly influenced the answer.
""".strip()


def build_judge_context_with_kg(patient_note, selected_relations):
    lines = ["Clinical context:", patient_note.strip(), "", "Retrieved knowledge graph information:"]

    if selected_relations:
        for rel in selected_relations:
            line = f"- {rel['source_name']} --{rel['type']}--> {rel['target_name']}"
            if rel.get("description"):
                line += f" | {rel['description']}"
            lines.append(line)
    else:
        lines.append("None")

    return "\n".join(lines).strip()


def build_judge_context_no_kg(patient_note):
    return f"""
Clinical context:
{patient_note.strip()}

Retrieved knowledge graph information:
None
""".strip()


def build_prompts(prompt_types, payload, patient_note):
    all_prompts = {
        "kg_full": build_relation_payload_message(payload),
        "no_kg": build_no_kg_prompt(patient_note),
        "kg_weighted": build_relation_payload_message_kg_weighted(payload),
    }
    return {key: all_prompts[key] for key in prompt_types if key in all_prompts}


def judge_context_for_output(output_key, patient_note, payload):
    if output_key == "no_kg":
        return build_judge_context_no_kg(patient_note), None

    return build_judge_context_with_kg(
        patient_note,
        payload["selected_relations"],
    ), "retrieval"


def summarize_judge_result(result):
    if not result:
        return None

    summary = {}

    for key, value in result.items():
        if isinstance(value, dict) and "accepted" in value:
            summary[key] = {
                "accepted": value.get("accepted"),
                "score": value.get("score"),
                "reasoning": value.get("reasoning"),
            }

    return summary


def build_test_assessment_summary(
    retrieval_judge_result,
    judge_results_by_output,
    case_meta_results_by_output,
    kg_comparison_judge_result=None,
):
    output_assessments = {}

    for key, judge_result in (judge_results_by_output or {}).items():
        output_assessments[key] = {
            "label": PROMPT_LABELS.get(key, key),
            "output_judges": summarize_judge_result(judge_result),
            "case_meta_judge": summarize_judge_result(
                case_meta_results_by_output.get(key)
                if case_meta_results_by_output
                else None
            ),
        }

    return {
        "retrieval_judges": summarize_judge_result(retrieval_judge_result),
        "output_assessments": output_assessments,
        "kg_comparison_judges": kg_comparison_judge_result,
    }

def score_above_zero(item):
    value = item.get("llm_relevance")
    if value is None:
        value = item.get("score", 0)

    try:
        return float(value) > 0
    except Exception:
        return False


def select_score_above_zero(rows):
    return [row for row in rows or [] if score_above_zero(row)]


def get_settings(args):
    return {
        "model_name": args.model,
        "judge_model": args.judge_model,
        "prompt_types": args.prompt_types,
        "run_judges": args.run_judges,
        "debug_judge_prompts": args.debug_judge_prompts,
        "kg_expansion_mode": args.kg_expansion_mode,
        "multi_hop_depth": args.multi_hop_depth,
        "max_multihop_candidates": args.max_multihop_candidates,
        "relation_rerank_top_k": args.relation_rerank_top_k,
        "penalize_hop_distance": args.penalize_hop_distance,
        "hop_weights": {
            1: args.hop_1_weight,
            2: args.hop_2_weight,
            3: args.hop_3_weight,
        },
        "diagnosis_retrieval_mode": args.diagnosis_retrieval_mode,
        "multi_term_per_term_top_k": args.multi_term_per_term_top_k,
        "multi_term_final_top_k": args.multi_term_final_top_k,
        "diagnosis_retrieval_weights": {
            "symptoms": args.weight_symptoms,
            "findings": args.weight_findings,
            "test_or_measurement_terms": args.weight_tests,
            "anatomy_terms": args.weight_anatomy,
            "possible_diagnoses_if_supported": args.weight_diagnoses,
            "clinically_relevant_risk_or_context_terms": args.weight_context,
            "lexical_variants_and_aliases": args.weight_aliases,
            "fallback_terms": args.weight_fallback_terms,
        },
        "selection_logic": {
            "diagnosis_nodes": "include rows where llm_relevance > 0, falling back to score > 0",
            "relations": "include rows where llm_relevance > 0, falling back to score > 0",
        },
    }


def build_simple_judge_overview(
    retrieval_judge_result,
    judge_results_by_output,
    case_meta_results_by_output,
    kg_comparison_judge_result=None,
):
    lines = []

    lines.append("JUDGE OVERVIEW")
    lines.append("=" * 80)

    if retrieval_judge_result:
        lines.append("")
        lines.append("Retrieval judges")

        for judge_name, judge_data in retrieval_judge_result.items():
            if isinstance(judge_data, dict) and "accepted" in judge_data:
                accepted = "PASS" if judge_data.get("accepted") else "FAIL"
                score = judge_data.get("score")
                reasoning = judge_data.get("reasoning", "")
                lines.append(f"- {judge_name}: {accepted} | score={score} | {reasoning}")
    else:
        lines.append("")
        lines.append("Retrieval judges: Not run or no result")

    if judge_results_by_output:
        lines.append("")
        lines.append("Output judges")

        for output_key, judge_result in judge_results_by_output.items():
            label = PROMPT_LABELS.get(output_key, output_key)
            lines.append("")
            lines.append(f"{label} ({output_key})")

            for judge_name, judge_data in judge_result.items():
                if isinstance(judge_data, dict) and "accepted" in judge_data:
                    accepted = "PASS" if judge_data.get("accepted") else "FAIL"
                    score = judge_data.get("score")
                    reasoning = judge_data.get("reasoning", "")
                    lines.append(f"- {judge_name}: {accepted} | score={score} | {reasoning}")
    else:
        lines.append("")
        lines.append("Output judges: Not run or no result")

    if case_meta_results_by_output:
        lines.append("")
        lines.append("Case meta judges")

        for output_key, meta_result in case_meta_results_by_output.items():
            label = PROMPT_LABELS.get(output_key, output_key)
            lines.append("")
            lines.append(f"{label} ({output_key})")

            for judge_name, judge_data in meta_result.items():
                if isinstance(judge_data, dict) and "accepted" in judge_data:
                    accepted = "PASS" if judge_data.get("accepted") else "FAIL"
                    score = judge_data.get("score")
                    reasoning = judge_data.get("reasoning", "")
                    lines.append(f"- {judge_name}: {accepted} | score={score} | {reasoning}")
    else:
        lines.append("")
        lines.append("Case meta judges: Not run or no result")
    
    if kg_comparison_judge_result and kg_comparison_judge_result.get("comparisons"):
        lines.append("")
        lines.append("KG comparison judges")

        for output_key, comparison in kg_comparison_judge_result["comparisons"].items():
            label = PROMPT_LABELS.get(output_key, output_key)
            lines.append("")
            lines.append(f"{label} ({output_key})")

            improvement = comparison.get("kg_improved_over_no_kg")
            if isinstance(improvement, dict):
                accepted = "PASS" if improvement.get("accepted") else "FAIL"
                score = improvement.get("score")
                reasoning = improvement.get("reasoning", "")
                lines.append(
                    f"- kg_improved_over_no_kg: {accepted} | score={score} | {reasoning}"
                )

            visible = comparison.get("kg_visible_influence")
            if isinstance(visible, dict):
                accepted = "PASS" if visible.get("accepted") else "FAIL"
                score = visible.get("score")
                reasoning = visible.get("reasoning", "")
                lines.append(
                    f"- kg_visible_influence: {accepted} | score={score} | {reasoning}"
                )
    else:
        lines.append("")
        lines.append("KG comparison judges: Not run or no result")

    return "\n".join(lines).strip()


def write_retrieval_judge_debug_file(
    item_result,
    patient_note,
    control_label,
    diagnosis_result_sets,
    payload,
    base_url,
):
    debug_dir = JUDGE_DEBUG_DIR / item_result["item_id"]

    all_node_candidates_text = format_diagnosis_candidates_for_label_match(
        diagnosis_result_sets,
        max_items_per_set=50,
    )

    selected_relations_text = format_kg_relations_for_prompt(payload)

    selected_relations_json = json.dumps(
        safe_json(payload.get("selected_relations", [])),
        ensure_ascii=False,
        indent=2,
    )

    debug_text = "\n\n".join([
        "=== ITEM ID ===",
        item_result["item_id"],
        "=== WORKER BASE URL ===",
        str(base_url),
        "=== PATIENT NOTE / input_text ===",
        patient_note,
        "=== CONTROL LABEL / expected ===",
        control_label,
        "=== ALL NODE CANDIDATES TEXT / output passed to control_label_match when all_node_candidates_text is provided ===",
        all_node_candidates_text,
        "=== SELECTED RELATIONS TEXT / selected KG payload text ===",
        selected_relations_text,
        "=== SELECTED RELATIONS JSON ===",
        selected_relations_json,
    ])

    atomic_write_text(
        debug_dir / "retrieval_control_label_match_INPUT.txt",
        debug_text,
    )


def run_batch_item(file_path, row_number, row, vector_data, embedder, settings, base_url=None, neo4j_driver=None):
    patient_note = str(row.get("pso_note", "") or "").strip()
    control_label = str(row.get("diagnosis", "") or "").strip()

    result = {
        "item_id": item_id(file_path, row_number, patient_note),
        "source": {
            "file": str(file_path),
            "file_name": file_path.name,
            "row_number": row_number,
            "row_data": row.to_dict(),
        },
        "timestamps": {
            "started_at": now_iso(),
            "finished_at": None,
        },
        "settings": settings,
        "worker": {
            "base_url": base_url,
        },
        "input": {
            "pso_note": patient_note,
            "diagnosis": control_label,
        },
        "judge_text_overview": None,
        "test_assessment_summary": None,
        "errors": [],
        "skipped": False,
    }

    if not patient_note:
        result["skipped"] = True
        result["errors"].append("Missing pso_note")
        result["timestamps"]["finished_at"] = now_iso()
        return result

    relationship_index = vector_data["relationship_index"]
    relationship_embeddings = vector_data["relationship_embeddings"]
    relationship_lexical = vector_data["relationship_lexical"]
    relationship_id_to_index = vector_data["relationship_id_to_index"]
    diagnosis_index = vector_data["diagnosis_index"]
    diagnosis_embeddings = vector_data["diagnosis_embeddings"]
    diagnosis_lexical = vector_data["diagnosis_lexical"]

    try:
        try:
            structured_query = extract_structured_query(
                patient_note,
                model_name=settings["model_name"],
                base_url=base_url,
            )
            structured_query_source = "llm"
        except Exception as e:
            structured_query = fallback_query_enhancement(patient_note)
            structured_query_source = "fallback"
            result["errors"].append(f"Structured query fallback used: {e}")

        enhanced_query = build_diagnosis_retrieval_query_from_structured(structured_query)

        diagnosis_result_sets = retrieve_diagnosis_candidates(
            query=enhanced_query,
            diagnosis_index=diagnosis_index,
            diagnosis_embeddings=diagnosis_embeddings,
            diagnosis_lexical=diagnosis_lexical,
            model=embedder,
            retrieval_mode=settings["diagnosis_retrieval_mode"],
            structured_query=structured_query,
            top_k=settings["multi_term_final_top_k"],
            per_term_top_k=settings["multi_term_per_term_top_k"],
            weights=settings["diagnosis_retrieval_weights"],
        )

        reranked_diagnosis_result_sets = {}
        selected_diagnosis_result_sets = {}
        diagnosis_rerank_artifacts = {}

        for search_name, candidates in diagnosis_result_sets.items():
            try:
                rerank = rerank_diagnosis_candidates(
                    patient_note=patient_note,
                    candidates=candidates,
                    model_name=settings["model_name"],
                    base_url=base_url,
                )
                reranked = rerank["all_candidates"]
                selected = select_score_above_zero(reranked)
                diagnosis_rerank_artifacts[search_name] = {
                    "prompt": rerank["prompt"],
                    "raw_response": rerank["raw_response"],
                }
            except Exception as e:
                reranked = candidates
                selected = select_score_above_zero(reranked)
                diagnosis_rerank_artifacts[search_name] = {
                    "prompt": None,
                    "raw_response": None,
                    "error": str(e),
                }
                result["errors"].append(
                    f"Diagnosis reranking fallback used for {search_name}: {e}"
                )

            reranked_diagnosis_result_sets[search_name] = reranked
            selected_diagnosis_result_sets[search_name] = selected

        selected_node_ids = []
        seen_node_ids = set()

        for selected_rows in selected_diagnosis_result_sets.values():
            for item in selected_rows:
                node_id = item.get("node_id")
                if node_id is not None and node_id not in seen_node_ids:
                    seen_node_ids.add(node_id)
                    selected_node_ids.append(node_id)

        if neo4j_driver is None:
            raise ValueError("Neo4j driver is missing.")

        relation_depths_by_relationship_id = collect_relation_candidates_from_neo4j(
            neo4j_driver=neo4j_driver,
            selected_node_ids=set(selected_node_ids),
            max_depth=settings["multi_hop_depth"],
            max_candidates=settings["max_multihop_candidates"],
        )

        relation_depths = {
            relationship_id_to_index[relationship_id]: hop_depth
            for relationship_id, hop_depth in relation_depths_by_relationship_id.items()
            if relationship_id in relationship_id_to_index
        }

        candidate_indices = list(relation_depths.keys())

        if candidate_indices:
            candidate_items_raw = [relationship_index[i] for i in candidate_indices]
            candidate_items = add_hop_depth_to_candidate_items(
                candidate_items=candidate_items_raw,
                candidate_indices=candidate_indices,
                relation_depths=relation_depths,
            )

            candidate_embeddings = relationship_embeddings[candidate_indices]
            candidate_lexical = {
                "tokenized_docs": [
                    relationship_lexical["tokenized_docs"][i]
                    for i in candidate_indices
                ],
                "idf": relationship_lexical["idf"],
            }

            expanded_relations = hybrid_rank(
                query=enhanced_query,
                index_items=candidate_items,
                embeddings=candidate_embeddings,
                lexical_data=candidate_lexical,
                model=embedder,
                alpha=0.75,
                beta=0.25,
                top_k=None,
            )

            expanded_relations = apply_hop_distance_penalty(
                expanded_relations,
                enabled=settings["penalize_hop_distance"],
                hop_weights=settings["hop_weights"],
            )
        else:
            expanded_relations = []

        relation_candidates_for_llm = expanded_relations[
            :settings["relation_rerank_top_k"]
        ]

        try:
            relation_rerank = rerank_relation_candidates(
                patient_note=patient_note,
                candidates=relation_candidates_for_llm,
                model_name=settings["model_name"],
                base_url=base_url,
            )
            reranked_relation_results = relation_rerank["all_candidates"]
            selected_relation_results = select_score_above_zero(
                reranked_relation_results
            )
            relation_rerank_prompt = relation_rerank["prompt"]
            relation_rerank_raw_response = relation_rerank["raw_response"]
        except Exception as e:
            reranked_relation_results = relation_candidates_for_llm
            selected_relation_results = select_score_above_zero(
                reranked_relation_results
            )
            relation_rerank_prompt = None
            relation_rerank_raw_response = None
            result["errors"].append(f"Relation reranking fallback used: {e}")

        payload = build_relation_payload(patient_note, selected_relation_results)
        diagnosis_prompts = build_prompts(
            settings["prompt_types"],
            payload,
            patient_note,
        )

        diagnosis_outputs = {}

        with ThreadPoolExecutor(max_workers=max(1, len(diagnosis_prompts))) as executor:
            futures = {
                executor.submit(
                    generate_llm_response,
                    prompt,
                    settings["model_name"],
                    base_url,
                ): key
                for key, prompt in diagnosis_prompts.items()
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    diagnosis_outputs[key] = future.result()
                except Exception as e:
                    diagnosis_outputs[key] = None
                    result["errors"].append(
                        f"Diagnosis output failed for {key}: {e}"
                    )

        retrieval_judge_result = None
        judge_results_by_output = {}
        case_meta_results_by_output = {}
        kg_comparison_judge_result = None

        if settings["run_judges"]:
            if not JUDGES_AVAILABLE:
                result["errors"].append(f"Judges unavailable: {JUDGE_IMPORT_ERROR}")
            else:
                if payload["selected_relations"]:
                    try:
                        all_node_candidates_text = format_diagnosis_candidates_for_label_match(
                            diagnosis_result_sets,
                            max_items_per_set=50,
                        )

                        if settings.get("debug_judge_prompts"):
                            write_retrieval_judge_debug_file(
                                item_result=result,
                                patient_note=patient_note,
                                control_label=control_label,
                                diagnosis_result_sets=diagnosis_result_sets,
                                payload=payload,
                                base_url=base_url,
                            )

                        retrieval_judge_result = run_retrieval_judges(
                            input_text=patient_note,
                            selected_relations=payload["selected_relations"],
                            model_name=settings["judge_model"],
                            control_label=control_label,
                            all_node_candidates_text=all_node_candidates_text,
                            base_url=base_url,
                        )
                    except Exception as e:
                        result["errors"].append(f"Retrieval judge failed: {e}")

                for output_key, output_text in diagnosis_outputs.items():
                    if not output_text:
                        continue

                    try:
                        judge_context, retrieval_marker = judge_context_for_output(
                            output_key,
                            patient_note,
                            payload,
                        )

                        output_judge = run_output_judges(
                            input_text=judge_context,
                            output_text=output_text,
                            model_name=settings["judge_model"],
                            expected_text=control_label,
                            base_url=base_url,
                        )

                        meta_judge = run_case_meta_judge(
                            input_text=judge_context,
                            output_text=output_text,
                            model_name=settings["judge_model"],
                            retrieval_results=(
                                retrieval_judge_result if retrieval_marker else None
                            ),
                            output_results=output_judge,
                            base_url=base_url,
                        )

                        judge_results_by_output[output_key] = output_judge
                        case_meta_results_by_output[output_key] = meta_judge

                    except Exception as e:
                        result["errors"].append(
                            f"Output judge failed for {output_key}: {e}"
                        )
                try:
                    no_kg_output = diagnosis_outputs.get("no_kg")

                    kg_outputs_by_key = {
                        key: value
                        for key, value in diagnosis_outputs.items()
                        if key != "no_kg" and value
                    }

                    if no_kg_output and kg_outputs_by_key:
                        kg_context = format_kg_relations_for_prompt(payload)

                        kg_comparison_judge_result = run_kg_comparison_judges(
                            patient_note=patient_note,
                            kg_context=kg_context,
                            kg_outputs_by_key=kg_outputs_by_key,
                            no_kg_output=no_kg_output,
                            model_name=settings["judge_model"],
                            control_label=control_label,
                            judge_results_by_output=judge_results_by_output,
                            base_url=base_url,
                        )

                except Exception as e:
                    result["errors"].append(f"KG comparison judges failed: {e}")

        result["judge_text_overview"] = build_simple_judge_overview(
            retrieval_judge_result=retrieval_judge_result,
            judge_results_by_output=judge_results_by_output,
            case_meta_results_by_output=case_meta_results_by_output,
            kg_comparison_judge_result=kg_comparison_judge_result,
        )

        result["test_assessment_summary"] = build_test_assessment_summary(
            retrieval_judge_result=retrieval_judge_result,
            judge_results_by_output=judge_results_by_output,
            case_meta_results_by_output=case_meta_results_by_output,
            kg_comparison_judge_result=kg_comparison_judge_result,
        )

        result.update({
            "query_processing": {
                "structured_query_source": structured_query_source,
                "structured_query": structured_query,
                "enhanced_query": enhanced_query,
            },
            "retrieval": {
                "diagnosis_result_sets": diagnosis_result_sets,
                "reranked_diagnosis_result_sets": reranked_diagnosis_result_sets,
                "selected_diagnosis_result_sets": selected_diagnosis_result_sets,
                "selected_node_ids": selected_node_ids,
                "relation_depths": relation_depths,
                "expanded_relations": expanded_relations,
                "reranked_relation_results": reranked_relation_results,
                "selected_relation_results": selected_relation_results,
            },
            "prompts": {
                "diagnosis_prompts": diagnosis_prompts,
                "diagnosis_rerank_artifacts": diagnosis_rerank_artifacts,
                "relation_rerank_prompt": relation_rerank_prompt,
                "relation_rerank_raw_response": relation_rerank_raw_response,
            },
            "payload": payload,
            "outputs": {
                "diagnosis_outputs": diagnosis_outputs,
            },
            "judges": {
                "retrieval_judge_result": retrieval_judge_result,
                "judge_results_by_output": judge_results_by_output,
                "case_meta_results_by_output": case_meta_results_by_output,
                "kg_comparison_judge_result": kg_comparison_judge_result,
            },
        })

    except Exception as e:
        result["errors"].append(f"Pipeline failed: {e}")

    result["timestamps"]["finished_at"] = now_iso()
    return result


def iter_excel_rows(input_dir):
    files = sorted(list(input_dir.glob("*.xlsx")) + list(input_dir.glob("*.xls")))

    for file_path in files:
        df = pd.read_excel(file_path)

        required = {"pso_note", "diagnosis"}
        missing = required - set(df.columns)

        if missing:
            yield file_path, None, None, f"Missing columns: {sorted(missing)}"
            continue

        for row_number, row in df.iterrows():
            yield file_path, row_number + 2, row, None


def run_batch(args):
    settings = get_settings(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if settings.get("debug_judge_prompts"):
        JUDGE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    vector_data = load_vector_data()
    embedder = SentenceTransformer(MODEL_NAME)
    neo4j_driver = create_neo4j_driver()

    worker_base_urls = BASE_URLS

    if not worker_base_urls:
        raise ValueError("No local model BASE_URL values found. Set BASE_URL1, BASE_URL2, or BASE_URL3.")

    if args.max_workers is None:
        max_workers = len(worker_base_urls)
    else:
        max_workers = min(len(worker_base_urls), max(1, args.max_workers))

    manifest = {
        "started_at": now_iso(),
        "finished_at": None,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "judge_debug_dir": str(JUDGE_DEBUG_DIR) if settings.get("debug_judge_prompts") else None,
        "settings": settings,
        "worker_base_urls": worker_base_urls,
        "max_workers": max_workers,
        "items": [],
        "errors": [],
    }

    jobs = []

    for file_path, row_number, row, error in iter_excel_rows(INPUT_DIR):
        if error:
            manifest["errors"].append({"file": str(file_path), "error": error})
            print(f"SKIP FILE {file_path.name}: {error}")
            continue

        patient_note = str(row.get("pso_note", "") or "").strip()
        current_item_id = item_id(file_path, row_number, patient_note)
        output_path = OUTPUT_DIR / f"{current_item_id}.json"

        if output_path.exists() and not args.force:
            print(f"SKIP existing {output_path.name}")
            manifest["items"].append({
                "item_id": current_item_id,
                "output_path": str(output_path),
                "skipped_existing": True,
            })
            continue

        jobs.append({
            "file_path": file_path,
            "row_number": row_number,
            "row": row,
            "item_id": current_item_id,
            "output_path": output_path,
        })

    def run_job(job_index, job):
        base_url = worker_base_urls[job_index % len(worker_base_urls)]

        print(
            f"RUN {job['file_path'].name} row {job['row_number']} "
            f"on {base_url or 'local'} -> {job['output_path'].name}"
        )

        result = run_batch_item(
            file_path=job["file_path"],
            row_number=job["row_number"],
            row=job["row"],
            vector_data=vector_data,
            embedder=embedder,
            settings=settings,
            base_url=base_url,
            neo4j_driver=neo4j_driver,
        )

        result["worker"] = {
            "base_url": base_url,
            "job_index": job_index,
        }

        atomic_write_json(job["output_path"], result)

        return {
            "item_id": job["item_id"],
            "output_path": str(job["output_path"]),
            "skipped_existing": False,
            "errors": result.get("errors", []),
            "test_assessment_summary": result.get("test_assessment_summary"),
            "worker": result.get("worker"),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_job, idx, job): job
            for idx, job in enumerate(jobs)
        }

        for future in as_completed(futures):
            job = futures[future]

            try:
                item_summary = future.result()
                manifest["items"].append(item_summary)
                print(f"DONE {job['output_path'].name}")
            except Exception as e:
                manifest["items"].append({
                    "item_id": job["item_id"],
                    "output_path": str(job["output_path"]),
                    "skipped_existing": False,
                    "errors": [str(e)],
                })
                print(f"FAILED {job['output_path'].name}: {e}")

    manifest["finished_at"] = now_iso()
    manifest_path = OUTPUT_DIR / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    atomic_write_json(manifest_path, manifest)

    neo4j_driver.close()

    print(f"DONE. Manifest saved to {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)

    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=DEFAULT_PROMPT_TYPES,
        choices=DEFAULT_PROMPT_TYPES,
    )

    parser.add_argument("--run-judges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-judge-prompts", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--kg-expansion-mode", default=DEFAULT_KG_EXPANSION_MODE, choices=["multi-hop"])
    parser.add_argument("--multi-hop-depth", type=int, default=DEFAULT_MULTI_HOP_DEPTH)
    parser.add_argument("--max-multihop-candidates", type=int, default=DEFAULT_MAX_MULTIHOP_CANDIDATES)
    parser.add_argument("--relation-rerank-top-k", type=int, default=DEFAULT_RELATION_RERANK_TOP_K)

    parser.add_argument(
        "--penalize-hop-distance",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PENALIZE_HOP_DISTANCE,
    )
    parser.add_argument("--hop-1-weight", type=float, default=DEFAULT_HOP_1_WEIGHT)
    parser.add_argument("--hop-2-weight", type=float, default=DEFAULT_HOP_2_WEIGHT)
    parser.add_argument("--hop-3-weight", type=float, default=DEFAULT_HOP_3_WEIGHT)

    parser.add_argument(
        "--diagnosis-retrieval-mode",
        default=DEFAULT_DIAGNOSIS_RETRIEVAL_MODE,
        choices=DIAGNOSIS_RETRIEVAL_MODES,
    )
    parser.add_argument("--multi-term-per-term-top-k", type=int, default=DEFAULT_MULTI_TERM_PER_TERM_TOP_K)
    parser.add_argument("--multi-term-final-top-k", type=int, default=DEFAULT_MULTI_TERM_FINAL_TOP_K)

    parser.add_argument("--weight-symptoms", type=float, default=DEFAULT_WEIGHT_SYMPTOMS)
    parser.add_argument("--weight-findings", type=float, default=DEFAULT_WEIGHT_FINDINGS)
    parser.add_argument("--weight-tests", type=float, default=DEFAULT_WEIGHT_TESTS)
    parser.add_argument("--weight-anatomy", type=float, default=DEFAULT_WEIGHT_ANATOMY)
    parser.add_argument("--weight-diagnoses", type=float, default=DEFAULT_WEIGHT_DIAGNOSES)
    parser.add_argument("--weight-context", type=float, default=DEFAULT_WEIGHT_CONTEXT)
    parser.add_argument("--weight-aliases", type=float, default=DEFAULT_WEIGHT_ALIASES)
    parser.add_argument("--weight-fallback-terms", type=float, default=DEFAULT_WEIGHT_FALLBACK_TERMS)
    parser.add_argument("--max-workers", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    run_batch(parse_args())