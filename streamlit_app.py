import html
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_PATH))

from batch_testing import (
    MODEL_NAME,
    DEFAULT_LOCAL_MODEL,
    load_vector_data,
    extract_structured_query,
    fallback_query_enhancement,
    build_diagnosis_retrieval_query_from_structured,
    retrieve_diagnosis_candidates,
    rerank_diagnosis_candidates,
    collect_relation_candidates_from_selected_nodes,
    add_hop_depth_to_candidate_items,
    hybrid_rank,
    apply_hop_distance_penalty,
    rerank_relation_candidates,
    build_relation_payload,
    build_relation_payload_message_kg_weighted,
    generate_llm_response,
    safe_json,
    select_score_above_zero,
    DEFAULT_DIAGNOSIS_RETRIEVAL_MODE,
    DEFAULT_MULTI_TERM_PER_TERM_TOP_K,
    DEFAULT_MULTI_TERM_FINAL_TOP_K,
    DEFAULT_KG_EXPANSION_MODE,
    DEFAULT_MULTI_HOP_DEPTH,
    DEFAULT_MAX_MULTIHOP_CANDIDATES,
    DEFAULT_RELATION_RERANK_TOP_K,
    DEFAULT_PENALIZE_HOP_DISTANCE,
    DEFAULT_HOP_1_WEIGHT,
    DEFAULT_HOP_2_WEIGHT,
    DEFAULT_HOP_3_WEIGHT,
    DEFAULT_WEIGHT_SYMPTOMS,
    DEFAULT_WEIGHT_FINDINGS,
    DEFAULT_WEIGHT_TESTS,
    DEFAULT_WEIGHT_ANATOMY,
    DEFAULT_WEIGHT_DIAGNOSES,
    DEFAULT_WEIGHT_CONTEXT,
    DEFAULT_WEIGHT_ALIASES,
    DEFAULT_WEIGHT_FALLBACK_TERMS,
)


load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

st.set_page_config(
    page_title="Clinical KG Assistant",
    page_icon="🩺",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --app-content-width: 980px;
        --app-side-padding: 2rem;
        --app-subtle-border: rgba(128, 128, 128, 0.20);
    }

    section.main > div.block-container,
    .main .block-container,
    div[data-testid="stAppViewContainer"] section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"],
    div[data-testid="stMainBlockContainer"] {
        max-width: var(--app-content-width) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: var(--app-side-padding) !important;
        padding-right: var(--app-side-padding) !important;
        padding-top: 0.75rem !important;
        padding-bottom: 7rem !important;
    }

    div[class*="st-key-top_header"] {
        position: sticky !important;
        top: 0 !important;
        z-index: 1000 !important;
        background: var(--background-color) !important;
        padding: 0.85rem 0 0.75rem 0 !important;
        margin-bottom: 1rem !important;
        border-bottom: 1px solid var(--app-subtle-border) !important;
    }

    div[class*="st-key-top_header"] h1 {
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }

    div[class*="st-key-flag"] button {
        padding: 0.05rem 0.25rem !important;
        min-height: 1.15rem !important;
        height: 1.15rem !important;
        line-height: 1 !important;
        font-size: 0.68rem !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: inherit !important;
    }

    div[class*="st-key-flag"] button:hover {
        background: rgba(128, 128, 128, 0.08) !important;
        border: none !important;
        color: inherit !important;
    }

    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        left: 0 !important;
        right: 0 !important;
        transform: none !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stBottom"] > div > div,
    .stChatFloatingInputContainer > div {
        width: min(var(--app-content-width), calc(100vw - 2 * var(--app-side-padding))) !important;
        max-width: min(var(--app-content-width), calc(100vw - 2 * var(--app-side-padding))) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stBottomBlockContainer"] > div,
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.85rem !important;
        border-color: rgba(128, 128, 128, 0.18) !important;
        background: rgba(128, 128, 128, 0.035) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(128, 128, 128, 0.32) !important;
        background: rgba(128, 128, 128, 0.055) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
        row-gap: 0.45rem !important;
    }

    .candidate-title {
        font-size: 0.96rem;
        line-height: 1.25;
        font-weight: 650;
        margin-bottom: 0.15rem;
    }

    .relation-title {
        font-size: 0.91rem;
        line-height: 1.28;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }

    .score-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0.14rem 0.46rem;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(128, 128, 128, 0.16);
        border: 1px solid rgba(128, 128, 128, 0.16);
        white-space: nowrap;
    }

    .source-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.32rem;
        margin-top: 0.55rem;
        margin-bottom: 0.55rem;
        padding-bottom: 0.15rem;
    }

    .source-chip {
        display: inline-flex;
        border-radius: 999px;
        padding: 0.08rem 0.42rem;
        font-size: 0.70rem;
        line-height: 1.2;
        background: rgba(49, 130, 206, 0.12);
        border: 1px solid rgba(49, 130, 206, 0.18);
        text-decoration: none !important;
        max-width: 100%;
    }

    .muted-small {
        font-size: 0.72rem;
        opacity: 0.60;
        margin-top: 0.45rem;
        margin-bottom: 0.55rem;
        padding-bottom: 0.15rem;
    }

    .section-help {
        font-size: 0.86rem;
        opacity: 0.68;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
    }

    .selected-count {
        font-size: 0.84rem;
        opacity: 0.72;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
    }

    div[data-testid="stToggle"] label {
        min-height: 1.35rem !important;
    }

    div[role="dialog"] button[aria-label="Close"],
    div[data-testid="stDialog"] button[aria-label="Close"] {
        display: none !important;
    }

    @media (max-width: 900px) {
        :root {
            --app-side-padding: 1rem;
        }

        .candidate-title,
        .relation-title {
            font-size: 0.9rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_vector_data():
    return load_vector_data()


@st.cache_resource
def cached_embedder():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def cached_neo4j_driver():
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not username or not password:
        return None

    return GraphDatabase.driver(uri, auth=(username, password))


@st.cache_data(show_spinner=False)
def cached_node_source_urls(node_ids):
    node_ids = [int(node_id) for node_id in node_ids if node_id is not None]

    if not node_ids:
        return {}

    driver = cached_neo4j_driver()

    if driver is None:
        return {}

    database = os.getenv("NEO4J_DATABASE", "neo4j")

    try:
        with driver.session(database=database) as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE e.node_id IN $node_ids
                RETURN
                    e.node_id AS node_id,
                    e.source_urls AS source_urls
                """,
                node_ids=node_ids,
            )

            return {
                int(record["node_id"]): record["source_urls"] or []
                for record in rows
            }

    except Exception as e:
        st.session_state.errors.append(f"Could not load source URLs from Neo4j: {e}")
        return {}


LOG_DB_PATH = PROJECT_ROOT / "data" / "logs" / "log.db"


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def json_dumps_safe(value):
    return json.dumps(safe_json(value), ensure_ascii=False)


def init_log_db():
    LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(LOG_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                patient_note TEXT,
                structured_query_json TEXT,
                enhanced_query TEXT,
                diagnosis_result_sets_json TEXT,
                diagnosis_candidates_json TEXT,
                background_payload_json TEXT,
                selected_nodes_json TEXT,
                relation_candidates_json TEXT,
                selected_relations_json TEXT,
                final_payload_json TEXT,
                final_output TEXT,
                errors_json TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                label TEXT,
                data_json TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                created_at TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_id TEXT,
                item_label TEXT,
                feedback_text TEXT NOT NULL,
                item_json TEXT
            )
            """
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_request_id ON events(request_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_request_id ON feedback(request_id)")
        conn.commit()


def create_request_log(request_id, patient_note):
    init_log_db()
    timestamp = now_iso()

    with sqlite3.connect(LOG_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO requests (
                request_id,
                created_at,
                updated_at,
                patient_note,
                errors_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request_id,
                timestamp,
                timestamp,
                patient_note,
                json_dumps_safe([]),
            ),
        )
        conn.commit()


def update_request_log(request_id, **fields):
    if not request_id:
        return

    init_log_db()

    allowed_fields = {
        "patient_note",
        "structured_query_json",
        "enhanced_query",
        "diagnosis_result_sets_json",
        "diagnosis_candidates_json",
        "background_payload_json",
        "selected_nodes_json",
        "relation_candidates_json",
        "selected_relations_json",
        "final_payload_json",
        "final_output",
        "errors_json",
    }

    clean_fields = {key: value for key, value in fields.items() if key in allowed_fields}

    if not clean_fields:
        return

    clean_fields["updated_at"] = now_iso()

    assignments = ", ".join([f"{key} = ?" for key in clean_fields])
    values = list(clean_fields.values())
    values.append(request_id)

    with sqlite3.connect(LOG_DB_PATH) as conn:
        conn.execute(
            f"""
            UPDATE requests
            SET {assignments}
            WHERE request_id = ?
            """,
            values,
        )
        conn.commit()


def log_event(request_id, event_type, entity_type=None, entity_id=None, label=None, data=None):
    if not request_id:
        return

    init_log_db()

    with sqlite3.connect(LOG_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO events (
                request_id,
                created_at,
                event_type,
                entity_type,
                entity_id,
                label,
                data_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                now_iso(),
                event_type,
                entity_type,
                str(entity_id) if entity_id is not None else None,
                label,
                json_dumps_safe(data or {}),
            ),
        )
        conn.commit()


def save_item_feedback(request_id, item_type, item_id, item_label, feedback_text, item_json):
    init_log_db()

    with sqlite3.connect(LOG_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                request_id,
                created_at,
                item_type,
                item_id,
                item_label,
                feedback_text,
                item_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                now_iso(),
                item_type,
                str(item_id) if item_id is not None else None,
                item_label,
                feedback_text,
                json_dumps_safe(item_json or {}),
            ),
        )
        conn.commit()


def invisible_key_suffix(key):
    key_text = str(key)
    chars = ["\u200b", "\u200c", "\u200d", "\u2060"]
    suffix = ""

    for char in key_text:
        suffix += chars[ord(char) % len(chars)]

    return suffix


def feedback_button(item_type, item_id, item_label, item_json, key, in_form=False):
    help_text = "Flag this item and explain what is wrong."
    clicked = st.button("🚩", key=key, help=help_text)

    if clicked:
        st.session_state.feedback_target = {
            "item_type": item_type,
            "item_id": str(item_id) if item_id is not None else None,
            "item_label": item_label,
            "item_json": safe_json(item_json or {}),
            "feedback_key": str(uuid.uuid4()),
        }
        feedback_dialog()


def process_feedback_button(label, item_type, item_label, item_json, key):
    clicked = st.button(label, key=key, help="Flag this whole step and explain what went wrong.")

    if clicked:
        st.session_state.feedback_target = {
            "item_type": item_type,
            "item_id": None,
            "item_label": item_label,
            "item_json": safe_json(item_json or {}),
            "feedback_key": str(uuid.uuid4()),
        }
        feedback_dialog()


def render_section_header(title, feedback_label, item_type, item_label, item_json, key):
    left_col, right_col = st.columns([0.74, 0.26], gap="small")

    with left_col:
        st.subheader(title)

    with right_col:
        st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
        process_feedback_button(
            label=feedback_label,
            item_type=item_type,
            item_label=item_label,
            item_json=item_json,
            key=key,
        )


def render_score_and_feedback(item, item_type, item_id, item_label, item_json, key, in_form=False):
    score = get_score(item)

    if score is None:
        score_text = "Score: N/A"
    else:
        score_text = f"Score: {score:.2f}"

    left_col, right_col = st.columns([0.86, 0.14], gap="small")

    with left_col:
        st.markdown(
            "<div style='font-size: 0.76rem; opacity: 0.68; margin-top: -0.35rem; margin-bottom: 0.12rem;'>"
            f"({html.escape(score_text)})"
            "</div>",
            unsafe_allow_html=True,
        )

    with right_col:
        feedback_button(
            item_type=item_type,
            item_id=item_id,
            item_label=item_label,
            item_json=item_json,
            key=key,
            in_form=in_form,
        )


def clear_feedback_target():
    st.session_state.feedback_target = None


@st.dialog("Flag content")
def feedback_dialog():
    target = st.session_state.get("feedback_target")

    if not target:
        st.write("No feedback target selected.")
        if st.button("Close", key="feedback_dialog_close_empty"):
            clear_feedback_target()
            st.rerun()
        return

    feedback_key = target.get("feedback_key") or "default"

    st.caption(target.get("item_label") or "Selected item")

    feedback_text = st.text_area(
        "Explain what is wrong or should be improved",
        height=160,
        key=f"feedback_dialog_text_{feedback_key}",
    )

    button_cols = st.columns([0.42, 0.58], gap="small")

    with button_cols[0]:
        if st.button("Submit feedback", type="primary", key=f"submit_feedback_dialog_{feedback_key}"):
            if not feedback_text.strip():
                st.warning("Please write a short explanation first.")
                return

            save_item_feedback(
                request_id=st.session_state.get("request_id"),
                item_type=target.get("item_type"),
                item_id=target.get("item_id"),
                item_label=target.get("item_label"),
                feedback_text=feedback_text.strip(),
                item_json=target.get("item_json"),
            )

            log_event(
                request_id=st.session_state.get("request_id"),
                event_type="feedback_submitted",
                entity_type=target.get("item_type"),
                entity_id=target.get("item_id"),
                label=target.get("item_label"),
                data={
                    "feedback_text": feedback_text.strip(),
                    "item": target.get("item_json"),
                },
            )

            clear_feedback_target()
            st.success("Feedback saved.")
            st.rerun()

    with button_cols[1]:
        if st.button("Close", key=f"close_feedback_dialog_{feedback_key}"):
            clear_feedback_target()
            st.rerun()


def init_state():
    defaults = {
        "messages": [],
        "patient_note": None,
        "structured_query": None,
        "structured_query_source": None,
        "enhanced_query": None,
        "diagnosis_result_sets": None,
        "diagnosis_candidates": [],
        "background_payload": None,
        "background_check_done": False,
        "selected_candidates": [],
        "selected_node_ids": [],
        "relation_candidates": [],
        "selected_relations": [],
        "nodes_confirmed": False,
        "relations_confirmed": False,
        "final_output": None,
        "payload": None,
        "errors": [],
        "request_id": None,
        "feedback_target": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def build_settings():
    return {
        "model_name": DEFAULT_LOCAL_MODEL,
        "kg_expansion_mode": DEFAULT_KG_EXPANSION_MODE,
        "multi_hop_depth": DEFAULT_MULTI_HOP_DEPTH,
        "max_multihop_candidates": DEFAULT_MAX_MULTIHOP_CANDIDATES,
        "relation_rerank_top_k": DEFAULT_RELATION_RERANK_TOP_K,
        "penalize_hop_distance": DEFAULT_PENALIZE_HOP_DISTANCE,
        "hop_weights": {
            1: DEFAULT_HOP_1_WEIGHT,
            2: DEFAULT_HOP_2_WEIGHT,
            3: DEFAULT_HOP_3_WEIGHT,
        },
        "diagnosis_retrieval_mode": DEFAULT_DIAGNOSIS_RETRIEVAL_MODE,
        "multi_term_per_term_top_k": DEFAULT_MULTI_TERM_PER_TERM_TOP_K,
        "multi_term_final_top_k": DEFAULT_MULTI_TERM_FINAL_TOP_K,
        "diagnosis_retrieval_weights": {
            "symptoms": DEFAULT_WEIGHT_SYMPTOMS,
            "findings": DEFAULT_WEIGHT_FINDINGS,
            "test_or_measurement_terms": DEFAULT_WEIGHT_TESTS,
            "anatomy_terms": DEFAULT_WEIGHT_ANATOMY,
            "possible_diagnoses_if_supported": DEFAULT_WEIGHT_DIAGNOSES,
            "clinically_relevant_risk_or_context_terms": DEFAULT_WEIGHT_CONTEXT,
            "lexical_variants_and_aliases": DEFAULT_WEIGHT_ALIASES,
            "fallback_terms": DEFAULT_WEIGHT_FALLBACK_TERMS,
        },
    }


def flatten_candidates(result_sets):
    rows = []
    seen = set()

    for search_name, candidates in result_sets.items():
        for item in candidates:
            node_id = item.get("node_id")

            if node_id in seen:
                continue

            seen.add(node_id)

            row = dict(item)
            row["search_source"] = search_name
            rows.append(row)

    rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return rows


def merge_llm_scores_into_diagnosis_candidates(diagnosis_candidates, scored_candidates):
    score_map = {}

    for item in scored_candidates or []:
        node_id = item.get("node_id")

        if node_id is None:
            continue

        existing = score_map.get(node_id)

        if existing is None:
            score_map[node_id] = item
            continue

        current_score = get_score(item)
        existing_score = get_score(existing)

        if (current_score or 0) > (existing_score or 0):
            score_map[node_id] = item

    enriched = []

    for candidate in diagnosis_candidates or []:
        node_id = candidate.get("node_id")
        row = dict(candidate)
        scored = score_map.get(node_id)

        if scored:
            for key in ["llm_relevance", "include_for_expansion", "payload_reason", "candidate_id"]:
                if key in scored:
                    row[key] = scored[key]

        enriched.append(row)

    enriched.sort(
        key=lambda x: (
            get_score(x) if get_score(x) is not None else -1,
            float(x.get("score") or 0),
        ),
        reverse=True,
    )

    return enriched


def node_label(candidate):
    name = str(candidate.get("name") or candidate.get("canonical_name") or "").strip()
    label = str(candidate.get("label") or "").strip()

    if name and label:
        return f"{name} ({label})"

    return name or label or str(candidate.get("node_id", "Unknown node"))


def relation_label(relation):
    source = str(relation.get("source_name") or "").strip()
    relation_type = str(relation.get("type") or "").strip()
    target = str(relation.get("target_name") or "").strip()

    return f"{source} — {relation_type} — {target}"


def get_score(item):
    for key in ["llm_relevance", "relevance_score", "score"]:
        value = item.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except Exception:
            continue

    return None


def render_score(item):
    score = get_score(item)

    if score is None:
        score_text = "Score: N/A"
    else:
        score_text = f"Score: {score:.2f}"

    st.markdown(
        "<div style='font-size: 0.76rem; opacity: 0.68; margin-top: -0.35rem; margin-bottom: 0.12rem;'>"
        f"({html.escape(score_text)})"
        "</div>",
        unsafe_allow_html=True,
    )


def node_source_urls(candidate, source_url_map):
    node_id = candidate.get("node_id")

    if node_id is None:
        return []

    return source_url_map.get(int(node_id), []) or []


def source_link_base_label(url):
    parsed = urlparse(str(url))
    path_parts = [part for part in parsed.path.split("/") if part]

    if not path_parts:
        return parsed.netloc or "source"

    return path_parts[-1]


def source_link_labels(urls):
    base_counts = {}
    labels = []

    for url in urls:
        base = source_link_base_label(url)
        base_counts[base] = base_counts.get(base, 0) + 1

        if base_counts[base] == 1:
            labels.append(base)
        else:
            labels.append(f"{base}{base_counts[base]}")

    return labels


def render_source_urls(urls):
    clean_urls = []
    seen = set()

    for url in urls or []:
        url = str(url).strip()

        if not url or url in seen:
            continue

        seen.add(url)
        clean_urls.append(url)

    if not clean_urls:
        st.markdown(
            "<div style='font-size: 0.72rem; opacity: 0.45; margin-top: 0.05rem; margin-bottom: 0.55rem;'>"
            "No source URL found."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    labels = source_link_labels(clean_urls)
    links = []

    for label, url in zip(labels, clean_urls):
        links.append(
            f"<a href='{html.escape(url, quote=True)}' target='_blank' "
            "style='opacity: 0.72; text-decoration: none;'>"
            f"{html.escape(label)}</a>"
        )

    st.markdown(
        "<div style='font-size: 0.72rem; line-height: 1.15; margin-top: 0.05rem; margin-bottom: 0.55rem;'>"
        + ", ".join(links)
        + "</div>",
        unsafe_allow_html=True,
    )



def compact_score_text(item):
    score = get_score(item)

    if score is None:
        return "N/A"

    return f"{score:.2f}"


def clean_source_urls(urls):
    clean_urls = []
    seen = set()

    for url in urls or []:
        url = str(url).strip()

        if not url or url in seen:
            continue

        seen.add(url)
        clean_urls.append(url)

    return clean_urls


def source_chips_html(urls, max_visible=3):
    clean_urls = clean_source_urls(urls)

    if not clean_urls:
        return "<div class='muted-small'>No source URL found.</div>"

    visible_urls = clean_urls[:max_visible]
    labels = source_link_labels(visible_urls)
    links = []

    for label, url in zip(labels, visible_urls):
        links.append(
            f"<a class='source-chip' href='{html.escape(url, quote=True)}' target='_blank'>"
            f"{html.escape(label)}</a>"
        )

    remaining = len(clean_urls) - len(visible_urls)

    if remaining > 0:
        links.append(f"<span class='source-chip'>+{remaining} more</span>")

    return "<div class='source-row'>" + "".join(links) + "</div>"


def render_score_pill(item):
    st.markdown(
        f"<span class='score-pill'>Score {html.escape(compact_score_text(item))}</span>",
        unsafe_allow_html=True,
    )


def render_node_candidate_card(candidate, idx, source_url_map, default_selected):
    candidate_label = node_label(candidate)

    with st.container(border=True):
        top_cols = st.columns([0.10, 0.64, 0.18, 0.08], gap="small")

        with top_cols[0]:
            selected = st.toggle(
                "Select node",
                value=default_selected,
                key=f"node_candidate_{idx}",
                label_visibility="collapsed",
            )

        with top_cols[1]:
            st.markdown(
                f"<div class='candidate-title'>{html.escape(candidate_label)}</div>",
                unsafe_allow_html=True,
            )

        with top_cols[2]:
            render_score_pill(candidate)

        with top_cols[3]:
            feedback_button(
                item_type="node",
                item_id=candidate.get("node_id"),
                item_label=candidate_label,
                item_json=candidate,
                key=f"flag_node_candidate_{candidate.get('node_id')}_{idx}",
            )

        st.markdown(
            source_chips_html(node_source_urls(candidate, source_url_map)),
            unsafe_allow_html=True,
        )

    return selected


def render_selected_node_card(candidate, idx, source_url_map):
    candidate_label = node_label(candidate)

    with st.container(border=True):
        top_cols = st.columns([0.74, 0.18, 0.08], gap="small")

        with top_cols[0]:
            st.markdown(
                f"<div class='candidate-title'>{html.escape(candidate_label)}</div>",
                unsafe_allow_html=True,
            )

        with top_cols[1]:
            render_score_pill(candidate)

        with top_cols[2]:
            feedback_button(
                item_type="node",
                item_id=candidate.get("node_id"),
                item_label=candidate_label,
                item_json=candidate,
                key=f"flag_selected_node_{candidate.get('node_id')}_{idx}",
            )

        st.markdown(
            source_chips_html(node_source_urls(candidate, source_url_map)),
            unsafe_allow_html=True,
        )


def render_relation_candidate_card(relation, idx, default_selected):
    rel_label = relation_label(relation)

    with st.container(border=True):
        top_cols = st.columns([0.10, 0.64, 0.18, 0.08], gap="small")

        with top_cols[0]:
            selected = st.toggle(
                "Select relation",
                value=default_selected,
                key=f"relation_candidate_{idx}",
                label_visibility="collapsed",
            )

        with top_cols[1]:
            st.markdown(
                f"<div class='relation-title'>{html.escape(rel_label)}</div>",
                unsafe_allow_html=True,
            )

        with top_cols[2]:
            render_score_pill(relation)

        with top_cols[3]:
            feedback_button(
                item_type="relation",
                item_id=relation.get("relationship_id"),
                item_label=rel_label,
                item_json=relation,
                key=f"flag_relation_candidate_{relation.get('relationship_id')}_{idx}",
            )

    return selected


def render_selected_relation_card(relation, idx):
    rel_label = relation_label(relation)

    with st.container(border=True):
        top_cols = st.columns([0.74, 0.18, 0.08], gap="small")

        with top_cols[0]:
            st.markdown(
                f"<div class='relation-title'>{html.escape(rel_label)}</div>",
                unsafe_allow_html=True,
            )

        with top_cols[1]:
            render_score_pill(relation)

        with top_cols[2]:
            feedback_button(
                item_type="relation",
                item_id=relation.get("relationship_id"),
                item_label=rel_label,
                item_json=relation,
                key=f"flag_selected_relation_{relation.get('relationship_id')}_{idx}",
            )


def run_node_search(patient_note, settings, vector_data, embedder):
    try:
        structured_query = extract_structured_query(
            patient_note,
            model_name=settings["model_name"],
        )
        structured_query_source = "llm"
    except Exception as e:
        structured_query = fallback_query_enhancement(patient_note)
        structured_query_source = "fallback"
        st.session_state.errors.append(f"Structured query fallback used: {e}")

    enhanced_query = build_diagnosis_retrieval_query_from_structured(structured_query)

    diagnosis_result_sets = retrieve_diagnosis_candidates(
        query=enhanced_query,
        diagnosis_index=vector_data["diagnosis_index"],
        diagnosis_embeddings=vector_data["diagnosis_embeddings"],
        diagnosis_lexical=vector_data["diagnosis_lexical"],
        model=embedder,
        retrieval_mode=settings["diagnosis_retrieval_mode"],
        structured_query=structured_query,
        top_k=settings["multi_term_final_top_k"],
        per_term_top_k=settings["multi_term_per_term_top_k"],
        weights=settings["diagnosis_retrieval_weights"],
    )

    diagnosis_candidates = flatten_candidates(diagnosis_result_sets)

    return structured_query_source, structured_query, enhanced_query, diagnosis_result_sets, diagnosis_candidates


def get_selected_node_ids(selected_candidates):
    selected_node_ids = []
    seen = set()

    for item in selected_candidates:
        node_id = item.get("node_id")

        if node_id is not None and node_id not in seen:
            seen.add(node_id)
            selected_node_ids.append(node_id)

    return selected_node_ids


def retrieve_relation_candidates(patient_note, selected_candidates, settings, vector_data, embedder, rerank=True):
    selected_node_ids = get_selected_node_ids(selected_candidates)

    relation_depths = collect_relation_candidates_from_selected_nodes(
        selected_node_ids=set(selected_node_ids),
        relationship_index=vector_data["relationship_index"],
        expansion_mode=settings["kg_expansion_mode"],
        multi_hop_depth=settings["multi_hop_depth"],
        max_multihop_candidates=settings["max_multihop_candidates"],
    )

    candidate_indices = list(relation_depths.keys())

    if not candidate_indices:
        return [], selected_node_ids

    candidate_items_raw = [
        vector_data["relationship_index"][i]
        for i in candidate_indices
    ]

    candidate_items = add_hop_depth_to_candidate_items(
        candidate_items=candidate_items_raw,
        candidate_indices=candidate_indices,
        relation_depths=relation_depths,
    )

    candidate_embeddings = vector_data["relationship_embeddings"][candidate_indices]

    candidate_lexical = {
        "tokenized_docs": [
            vector_data["relationship_lexical"]["tokenized_docs"][i]
            for i in candidate_indices
        ],
        "idf": vector_data["relationship_lexical"]["idf"],
    }

    expanded_relations = hybrid_rank(
        query=st.session_state.enhanced_query,
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

    relation_candidates = expanded_relations[: settings["relation_rerank_top_k"]]

    if not rerank:
        return relation_candidates, selected_node_ids

    try:
        relation_rerank = rerank_relation_candidates(
            patient_note=patient_note,
            candidates=relation_candidates,
            model_name=settings["model_name"],
        )

        relation_candidates = relation_rerank["all_candidates"]

    except Exception as e:
        st.session_state.errors.append(f"Relation reranking fallback used: {e}")

    return relation_candidates, selected_node_ids


def run_background_kg_check(patient_note, diagnosis_candidates, diagnosis_result_sets, settings, vector_data, embedder):
    selected_candidates = []
    all_scored_candidates = []

    for search_name, candidates in diagnosis_result_sets.items():
        try:
            rerank = rerank_diagnosis_candidates(
                patient_note=patient_note,
                candidates=candidates,
                model_name=settings["model_name"],
            )

            all_scored_candidates.extend(rerank["all_candidates"])
            selected_candidates.extend(select_score_above_zero(rerank["all_candidates"]))

        except Exception as e:
            st.session_state.errors.append(f"Background diagnosis reranking fallback used for {search_name}: {e}")
            selected_candidates.extend(candidates[:5])
            all_scored_candidates.extend(candidates[:5])

    unique_candidates = []
    seen = set()

    for item in selected_candidates:
        node_id = item.get("node_id")

        if node_id is None or node_id in seen:
            continue

        seen.add(node_id)
        unique_candidates.append(item)

    if not unique_candidates:
        unique_candidates = diagnosis_candidates[:5]

    relation_candidates, _ = retrieve_relation_candidates(
        patient_note=patient_note,
        selected_candidates=unique_candidates,
        settings=settings,
        vector_data=vector_data,
        embedder=embedder,
        rerank=True,
    )

    selected_relations = select_score_above_zero(relation_candidates)
    payload = build_relation_payload(patient_note, selected_relations)

    return payload, all_scored_candidates


def generate_final_answer(patient_note, selected_relations, settings):
    payload = build_relation_payload(patient_note, selected_relations)
    prompt = build_relation_payload_message_kg_weighted(payload)

    output = generate_llm_response(
        prompt=prompt,
        model_name=settings["model_name"],
    )

    return output, payload


def save_feedback(feedback_text):
    feedback_dir = PROJECT_ROOT / "data" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    feedback_path = feedback_dir / "retrieval_feedback.jsonl"

    record = {
        "patient_note": st.session_state.patient_note,
        "feedback": feedback_text,
        "background_payload": st.session_state.background_payload,
    }

    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_json(record), ensure_ascii=False) + "\n")


init_state()
init_log_db()

vector_data = cached_vector_data()
embedder = cached_embedder()
settings = build_settings()

top_header = st.container(key="top_header")

with top_header:
    st.title("Clinical KG Assistant")
    st.caption("Enter a patient note, choose relevant nodes, then choose relevant relations.")

    if cached_neo4j_driver() is None:
        st.warning("Neo4j connection settings are missing. Source URLs will not be shown.")


user_input = st.chat_input("Paste patient note here...")

if user_input:
    st.session_state.messages = [
        {
            "role": "user",
            "content": user_input,
        }
    ]

    st.session_state.patient_note = user_input
    st.session_state.structured_query = None
    st.session_state.structured_query_source = None
    st.session_state.enhanced_query = None
    st.session_state.diagnosis_result_sets = None
    st.session_state.diagnosis_candidates = []
    st.session_state.background_payload = None
    st.session_state.background_check_done = False
    st.session_state.selected_candidates = []
    st.session_state.selected_node_ids = []
    st.session_state.relation_candidates = []
    st.session_state.selected_relations = []
    st.session_state.nodes_confirmed = False
    st.session_state.relations_confirmed = False
    st.session_state.final_output = None
    st.session_state.payload = None
    st.session_state.errors = []
    st.session_state.feedback_target = None
    st.session_state.request_id = str(uuid.uuid4())

    create_request_log(
        request_id=st.session_state.request_id,
        patient_note=user_input,
    )
    log_event(
        request_id=st.session_state.request_id,
        event_type="request_started",
        data={"patient_note": user_input},
    )

    with st.spinner("Searching the knowledge graph..."):
        try:
            (
                structured_query_source,
                structured_query,
                enhanced_query,
                diagnosis_result_sets,
                diagnosis_candidates,
            ) = run_node_search(
                patient_note=user_input,
                settings=settings,
                vector_data=vector_data,
                embedder=embedder,
            )

            st.session_state.structured_query_source = structured_query_source
            st.session_state.structured_query = structured_query
            st.session_state.enhanced_query = enhanced_query
            st.session_state.diagnosis_result_sets = diagnosis_result_sets
            st.session_state.diagnosis_candidates = diagnosis_candidates

            background_payload, all_scored_candidates = run_background_kg_check(
                patient_note=user_input,
                diagnosis_candidates=diagnosis_candidates,
                diagnosis_result_sets=diagnosis_result_sets,
                settings=settings,
                vector_data=vector_data,
                embedder=embedder,
            )

            diagnosis_candidates = merge_llm_scores_into_diagnosis_candidates(
                diagnosis_candidates,
                all_scored_candidates,
            )

            st.session_state.diagnosis_candidates = diagnosis_candidates
            st.session_state.background_payload = background_payload
            st.session_state.background_check_done = True

            update_request_log(
                request_id=st.session_state.request_id,
                structured_query_json=json_dumps_safe(structured_query),
                enhanced_query=enhanced_query,
                diagnosis_result_sets_json=json_dumps_safe(diagnosis_result_sets),
                diagnosis_candidates_json=json_dumps_safe(diagnosis_candidates),
                background_payload_json=json_dumps_safe(background_payload),
                errors_json=json_dumps_safe(st.session_state.errors),
            )
            log_event(
                request_id=st.session_state.request_id,
                event_type="node_search_completed",
                data={
                    "structured_query_source": structured_query_source,
                    "enhanced_query": enhanced_query,
                    "diagnosis_candidate_count": len(diagnosis_candidates or []),
                },
            )

        except Exception as e:
            st.session_state.errors.append(f"Initial retrieval check failed: {e}")
            update_request_log(
                request_id=st.session_state.get("request_id"),
                errors_json=json_dumps_safe(st.session_state.errors),
            )
            log_event(
                request_id=st.session_state.get("request_id"),
                event_type="initial_retrieval_failed",
                data={"error": str(e)},
            )
            st.error(f"Initial retrieval check failed: {e}")
            st.stop()

    st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


all_visible_node_ids = []

for candidate in st.session_state.diagnosis_candidates:
    if candidate.get("node_id") is not None:
        all_visible_node_ids.append(int(candidate["node_id"]))

for candidate in st.session_state.selected_candidates:
    if candidate.get("node_id") is not None:
        all_visible_node_ids.append(int(candidate["node_id"]))

source_url_map = cached_node_source_urls(tuple(sorted(set(all_visible_node_ids))))


main_area = st.container()

with main_area:
    if st.session_state.selected_candidates:
        st.subheader("Selected nodes")
        st.markdown(
            "<div class='section-help'>Nodes currently included for relation expansion.</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2, gap="medium")

        for idx, candidate in enumerate(st.session_state.selected_candidates):
            with cols[idx % 2]:
                render_selected_node_card(
                    candidate=candidate,
                    idx=idx,
                    source_url_map=source_url_map,
                )

    if st.session_state.selected_relations:
        st.subheader("Selected relations")
        st.markdown(
            "<div class='section-help'>Relations currently included in the final answer payload.</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2, gap="medium")

        for idx, relation in enumerate(st.session_state.selected_relations):
            with cols[idx % 2]:
                render_selected_relation_card(relation=relation, idx=idx)

    if st.session_state.final_output:
        render_section_header(
            title="Weighted-KG answer",
            feedback_label="🚩 Flag output",
            item_type="final_output_step",
            item_label="Final Weighted-KG answer",
            item_json={
                "patient_note": st.session_state.patient_note,
                "selected_relations": st.session_state.selected_relations,
                "payload": st.session_state.payload,
                "final_output": st.session_state.final_output,
                "errors": st.session_state.errors,
            },
            key="process_feedback_final_output",
        )
        st.markdown(st.session_state.final_output)
        
        if st.session_state.errors:
            with st.expander("Warnings / fallback notes"):
                for error in st.session_state.errors:
                    st.write(error)

        st.stop()

    if st.session_state.patient_note and not st.session_state.diagnosis_candidates and not st.session_state.nodes_confirmed:
        render_section_header(
            title="Select relevant nodes",
            feedback_label="🚩 Flag node search",
            item_type="node_search_step",
            item_label="Node search returned no candidates",
            item_json={
                "patient_note": st.session_state.patient_note,
                "structured_query": st.session_state.structured_query,
                "structured_query_source": st.session_state.structured_query_source,
                "enhanced_query": st.session_state.enhanced_query,
                "diagnosis_result_sets": st.session_state.diagnosis_result_sets,
                "diagnosis_candidates": st.session_state.diagnosis_candidates,
                "candidate_count": 0,
                "errors": st.session_state.errors,
            },
            key="process_feedback_empty_node_search",
        )
        st.warning("No node candidates were found for this patient note.")
        st.stop()

    if st.session_state.diagnosis_candidates and not st.session_state.nodes_confirmed:
        render_section_header(
            title="Select relevant nodes",
            feedback_label="🚩 Flag node search",
            item_type="node_search_step",
            item_label="Node search / diagnosis candidate retrieval",
            item_json={
                "patient_note": st.session_state.patient_note,
                "structured_query": st.session_state.structured_query,
                "structured_query_source": st.session_state.structured_query_source,
                "enhanced_query": st.session_state.enhanced_query,
                "diagnosis_result_sets": st.session_state.diagnosis_result_sets,
                "diagnosis_candidates": st.session_state.diagnosis_candidates,
                "candidate_count": len(st.session_state.diagnosis_candidates or []),
                "errors": st.session_state.errors,
            },
            key="process_feedback_node_search",
        )
        st.markdown(
            "<div class='section-help'>Review the suggested nodes. Toggle the ones that should be used for relation retrieval. Source links are kept as compact chips inside each card.</div>",
            unsafe_allow_html=True,
        )

        selected_indices = []
        cols = st.columns(2, gap="medium")

        for idx, candidate in enumerate(st.session_state.diagnosis_candidates):
            with cols[idx % 2]:
                selected = render_node_candidate_card(
                    candidate=candidate,
                    idx=idx,
                    source_url_map=source_url_map,
                    default_selected=idx < 5,
                )

                if selected:
                    selected_indices.append(idx)

        st.markdown(
            f"<div class='selected-count'>{len(selected_indices)} node(s) selected</div>",
            unsafe_allow_html=True,
        )

        submitted = st.button(
            "Continue to relation selection",
            type="primary",
            key="continue_to_relation_selection",
        )

        if submitted:
            selected_candidates = [
                st.session_state.diagnosis_candidates[i]
                for i in selected_indices
            ]

            if not selected_candidates:
                st.error("Please select at least one node, or flag the node search if none of the suggestions are relevant.")
                process_feedback_button(
                    label="🚩 Flag node search",
                    item_type="node_search_step",
                    item_label="No relevant nodes found in node search",
                    item_json={
                        "patient_note": st.session_state.patient_note,
                        "structured_query": st.session_state.structured_query,
                        "enhanced_query": st.session_state.enhanced_query,
                        "diagnosis_candidates": st.session_state.diagnosis_candidates,
                        "candidate_count": len(st.session_state.diagnosis_candidates or []),
                        "selected_count": 0,
                        "errors": st.session_state.errors,
                    },
                    key="process_feedback_no_nodes_selected",
                )
                st.stop()

            with st.spinner("Retrieving relation candidates..."):
                relation_candidates, selected_node_ids = retrieve_relation_candidates(
                    patient_note=st.session_state.patient_note,
                    selected_candidates=selected_candidates,
                    settings=settings,
                    vector_data=vector_data,
                    embedder=embedder,
                    rerank=True,
                )

            st.session_state.selected_candidates = selected_candidates
            st.session_state.selected_node_ids = selected_node_ids
            st.session_state.relation_candidates = relation_candidates
            st.session_state.nodes_confirmed = True

            update_request_log(
                request_id=st.session_state.request_id,
                selected_nodes_json=json_dumps_safe(selected_candidates),
                relation_candidates_json=json_dumps_safe(relation_candidates),
                errors_json=json_dumps_safe(st.session_state.errors),
            )
            log_event(
                request_id=st.session_state.request_id,
                event_type="nodes_confirmed",
                data={
                    "selected_node_ids": selected_node_ids,
                    "selected_nodes": selected_candidates,
                    "relation_candidate_count": len(relation_candidates or []),
                },
            )

            st.rerun()

        st.stop()

    if st.session_state.nodes_confirmed and not st.session_state.relations_confirmed:
        render_section_header(
            title="Select relevant relations",
            feedback_label="🚩 Flag relation search",
            item_type="relation_search_step",
            item_label="Relation search / relation candidate retrieval",
            item_json={
                "patient_note": st.session_state.patient_note,
                "selected_nodes": st.session_state.selected_candidates,
                "selected_node_ids": st.session_state.selected_node_ids,
                "relation_candidates": st.session_state.relation_candidates,
                "candidate_count": len(st.session_state.relation_candidates or []),
                "errors": st.session_state.errors,
            },
            key="process_feedback_relation_search",
        )
        st.markdown(
            "<div class='section-help'>Review the relation candidates. Toggle the relations that should be included in the Weighted-KG answer.</div>",
            unsafe_allow_html=True,
        )

        if not st.session_state.relation_candidates:
            st.warning("No relation candidates were found from the selected nodes.")
            process_feedback_button(
                label="🚩 Flag relation search",
                item_type="relation_search_step",
                item_label="No relation candidates found",
                item_json={
                    "patient_note": st.session_state.patient_note,
                    "selected_nodes": st.session_state.selected_candidates,
                    "selected_node_ids": st.session_state.selected_node_ids,
                    "relation_candidates": st.session_state.relation_candidates,
                    "candidate_count": 0,
                    "errors": st.session_state.errors,
                },
                key="process_feedback_no_relation_candidates",
            )

        selected_relation_indices = []
        cols = st.columns(2, gap="medium")

        for idx, relation in enumerate(st.session_state.relation_candidates):
            default_selected = bool(relation.get("include_in_payload", False))

            if not default_selected:
                score = relation.get("llm_relevance")
                default_selected = float(score or 0) > 0

            with cols[idx % 2]:
                selected = render_relation_candidate_card(
                    relation=relation,
                    idx=idx,
                    default_selected=default_selected,
                )

                if selected:
                    selected_relation_indices.append(idx)

        st.markdown(
            f"<div class='selected-count'>{len(selected_relation_indices)} relation(s) selected</div>",
            unsafe_allow_html=True,
        )

        submitted = st.button(
            "Generate answer",
            type="primary",
            key="generate_answer",
        )

        if submitted:
            selected_relations = [
                st.session_state.relation_candidates[i]
                for i in selected_relation_indices
            ]

            if not selected_relations:
                st.error("Please select at least one relation, or flag the relation search if none of the suggestions are relevant.")
                process_feedback_button(
                    label="🚩 Flag relation search",
                    item_type="relation_search_step",
                    item_label="No relevant relations found in relation search",
                    item_json={
                        "patient_note": st.session_state.patient_note,
                        "selected_nodes": st.session_state.selected_candidates,
                        "selected_node_ids": st.session_state.selected_node_ids,
                        "relation_candidates": st.session_state.relation_candidates,
                        "candidate_count": len(st.session_state.relation_candidates or []),
                        "selected_count": 0,
                        "errors": st.session_state.errors,
                    },
                    key="process_feedback_no_relations_selected",
                )
                st.stop()

            with st.spinner("Generating Weighted-KG answer..."):
                try:
                    output, payload = generate_final_answer(
                        patient_note=st.session_state.patient_note,
                        selected_relations=selected_relations,
                        settings=settings,
                    )

                    st.session_state.selected_relations = selected_relations
                    st.session_state.payload = payload
                    st.session_state.final_output = output
                    st.session_state.relations_confirmed = True

                    update_request_log(
                        request_id=st.session_state.request_id,
                        selected_relations_json=json_dumps_safe(selected_relations),
                        final_payload_json=json_dumps_safe(payload),
                        final_output=output,
                        errors_json=json_dumps_safe(st.session_state.errors),
                    )
                    log_event(
                        request_id=st.session_state.request_id,
                        event_type="final_answer_generated",
                        data={
                            "selected_relations": selected_relations,
                            "payload": payload,
                            "output": output,
                        },
                    )

                except Exception as e:
                    st.session_state.errors.append(f"Weighted-KG generation failed: {e}")
                    update_request_log(
                        request_id=st.session_state.get("request_id"),
                        errors_json=json_dumps_safe(st.session_state.errors),
                    )
                    log_event(
                        request_id=st.session_state.get("request_id"),
                        event_type="final_answer_failed",
                        data={"error": str(e)},
                    )
                    st.error(f"Weighted-KG generation failed: {e}")
                    st.stop()

            st.rerun()

        st.stop()
