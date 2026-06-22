import argparse
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DIAGNOSIS_LABEL_GROUP = "DIAGNOSIS_LIKE"
TOKEN_PATTERN = re.compile(r"\b[\w\-æøåÆØÅ\.]+\b", re.UNICODE)


def tokenize(text):
    if not text:
        return []
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def fetch_relationships(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT
            r.relationship_id,
            r.source_node_id,
            s.canonical_name AS source_name,
            s.normalized_name AS source_normalized_name,
            COALESCE(s.canonical_label, s.label) AS source_label,
            COALESCE(s.label_group, 'OTHER') AS source_label_group,
            s.description AS source_description,
            r.target_node_id,
            t.canonical_name AS target_name,
            t.normalized_name AS target_normalized_name,
            COALESCE(t.canonical_label, t.label) AS target_label,
            COALESCE(t.label_group, 'OTHER') AS target_label_group,
            t.description AS target_description,
            r.relationship_type,
            r.description
        FROM relationships r
        JOIN nodes s ON r.source_node_id = s.node_id
        JOIN nodes t ON r.target_node_id = t.node_id
        ORDER BY r.relationship_id
    """)
    rows = cur.fetchall()

    relationships = []
    for row in rows:
        relationships.append({
            "relationship_id": row[0],
            "source_node_id": row[1],
            "source_name": row[2] or "",
            "source_normalized_name": row[3] or "",
            "source_label": row[4] or "",
            "source_label_group": row[5] or "OTHER",
            "source_description": row[6] or "",
            "target_node_id": row[7],
            "target_name": row[8] or "",
            "target_normalized_name": row[9] or "",
            "target_label": row[10] or "",
            "target_label_group": row[11] or "OTHER",
            "target_description": row[12] or "",
            "type": row[13] or "",
            "description": row[14] or "",
        })
    return relationships


def fetch_relationship_sources(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT
            rs.relationship_id,
            sf.filename
        FROM relationship_sources rs
        JOIN source_files sf
          ON rs.source_file_id = sf.source_file_id
        ORDER BY rs.relationship_id, sf.filename
    """)
    rows = cur.fetchall()

    source_map = {}
    for relationship_id, filename in rows:
        source_map.setdefault(relationship_id, []).append(filename)

    for relationship_id in source_map:
        source_map[relationship_id] = sorted(set(source_map[relationship_id]))

    return source_map


def fetch_diagnosis_nodes(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT
            node_id,
            canonical_name,
            normalized_name,
            COALESCE(canonical_label, label) AS label,
            COALESCE(label_group, 'OTHER') AS label_group,
            description
        FROM nodes
        WHERE COALESCE(label_group, 'OTHER') = ?
        ORDER BY node_id
    """, (DIAGNOSIS_LABEL_GROUP,))

    rows = cur.fetchall()

    nodes = []
    for row in rows:
        nodes.append({
            "node_id": row[0],
            "name": row[1] or "",
            "normalized_name": row[2] or "",
            "label": row[3] or "",
            "label_group": row[4] or "OTHER",
            "description": row[5] or "",
        })
    return nodes


def build_relationship_lookup(relationships):
    by_source = defaultdict(list)
    by_target = defaultdict(list)

    for rel in relationships:
        by_source[rel["source_node_id"]].append(rel)
        by_target[rel["target_node_id"]].append(rel)

    return by_source, by_target


def relationship_to_text(rel):
    parts = [
        f"{rel['source_name']} [{rel['source_label']} | {rel['source_label_group']}] "
        f"{rel['type']} "
        f"{rel['target_name']} [{rel['target_label']} | {rel['target_label_group']}]"
    ]

    if rel["description"]:
        parts.append(rel["description"])
    if rel["source_description"]:
        parts.append(f"Kilde: {rel['source_description']}")
    if rel["target_description"]:
        parts.append(f"Mål: {rel['target_description']}")

    return ". ".join(part.strip() for part in parts if part.strip())


def diagnosis_node_to_text(node, outgoing_rels, incoming_rels, max_relations=12):
    lines = [f"{node['name']} [{node['label']} | {node['label_group']}]"]

    if node["description"]:
        lines.append(node["description"])

    relation_lines = []
    preferred_groups = {
        "FINDING_LIKE",
        "ANATOMY",
        "RISK_OR_CAUSE",
        "TEST_OR_MEASUREMENT",
        "TREATMENT",
    }

    for rel in outgoing_rels:
        if rel.get("target_label_group") in preferred_groups:
            relation_lines.append(
                f"{rel['type']} {rel['target_name']} [{rel['target_label']} | {rel['target_label_group']}]"
                + (f": {rel['description']}" if rel["description"] else "")
            )

    for rel in incoming_rels:
        if rel.get("source_label_group") in preferred_groups:
            relation_lines.append(
                f"INVERSE {rel['type']} {rel['source_name']} [{rel['source_label']} | {rel['source_label_group']}]"
                + (f": {rel['description']}" if rel["description"] else "")
            )

    if relation_lines:
        lines.append("Relationer:")
        lines.extend(relation_lines[:max_relations])

    return "\n".join(lines)


def build_embeddings(texts, model_name, batch_size):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings


def build_lexical_metadata(docs):
    tokenized_docs = [tokenize(doc) for doc in docs]
    df_counter = Counter()

    for tokens in tokenized_docs:
        for token in set(tokens):
            df_counter[token] += 1

    n_docs = len(tokenized_docs)
    idf = {}
    for token, df in df_counter.items():
        idf[token] = math.log((1 + n_docs) / (1 + df)) + 1

    return {
        "tokenized_docs": tokenized_docs,
        "idf": idf,
        "doc_count": n_docs,
    }


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_sample_embeddings(
    relationships,
    relationship_texts,
    diagnosis_nodes,
    diagnosis_texts,
    num_samples=3,
):
    print("\n" + "=" * 80)
    print("SAMPLE RELATIONSHIP EMBEDDINGS")
    print("=" * 80)

    for i in range(min(num_samples, len(relationships))):
        rel = relationships[i]
        text = relationship_texts[i]

        print(f"\n--- Relationship Sample {i + 1} ---")
        print(f"Relationship ID: {rel['relationship_id']}")
        print(f"Type: {rel['type']}")
        print(f"Source: {rel['source_name']} [{rel['source_label']} | {rel['source_label_group']}]")
        print(f"Target: {rel['target_name']} [{rel['target_label']} | {rel['target_label_group']}]")
        print("\nEmbedding Text:\n")
        print(text)
        print("-" * 80)

    print("\n" + "=" * 80)
    print("SAMPLE DIAGNOSIS EMBEDDINGS")
    print("=" * 80)

    for i in range(min(num_samples, len(diagnosis_nodes))):
        node = diagnosis_nodes[i]
        text = diagnosis_texts[i]

        print(f"\n--- Diagnosis Sample {i + 1} ---")
        print(f"Node ID: {node['node_id']}")
        print(f"Name: {node['name']}")
        print(f"Label: {node['label']} | {node['label_group']}")
        print("\nEmbedding Text:\n")
        print(text)
        print("-" * 80)


def print_specific_diagnosis_samples(diagnosis_nodes, diagnosis_texts, names):
    if not names:
        return

    wanted = {name.strip().lower() for name in names if name.strip()}
    matches = []

    for node, text in zip(diagnosis_nodes, diagnosis_texts):
        name = (node.get("name") or "").strip().lower()
        if name in wanted:
            matches.append((node, text))

    print("\n" + "=" * 80)
    print("SPECIFIC DIAGNOSIS EMBEDDING SAMPLES")
    print("=" * 80)

    if not matches:
        print("No exact diagnosis node name matches found for the requested names.")
        return

    for i, (node, text) in enumerate(matches, start=1):
        print(f"\n--- Specific Diagnosis Sample {i} ---")
        print(f"Node ID: {node['node_id']}")
        print(f"Name: {node['name']}")
        print(f"Label: {node['label']} | {node['label_group']}")
        print("\nEmbedding Text:\n")
        print(text)
        print("-" * 80)


def save_outputs(
    output_dir,
    relationships,
    relationship_texts,
    relationship_embeddings,
    relationship_sources,
    diagnosis_nodes,
    diagnosis_texts,
    diagnosis_embeddings,
    model_name,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    relationship_index = []
    for rel, text in zip(relationships, relationship_texts):
        relationship_index.append({
            "id": f"rel:{rel['relationship_id']}",
            "relationship_id": rel["relationship_id"],
            "source_node_id": rel["source_node_id"],
            "source_name": rel["source_name"],
            "source_normalized_name": rel["source_normalized_name"],
            "source_label": rel["source_label"],
            "source_label_group": rel["source_label_group"],
            "source_description": rel["source_description"],
            "target_node_id": rel["target_node_id"],
            "target_name": rel["target_name"],
            "target_normalized_name": rel["target_normalized_name"],
            "target_label": rel["target_label"],
            "target_label_group": rel["target_label_group"],
            "target_description": rel["target_description"],
            "type": rel["type"],
            "description": rel["description"],
            "sources": relationship_sources.get(rel["relationship_id"], []),
            "embedding_text": text,
        })

    diagnosis_index = []
    for node, text in zip(diagnosis_nodes, diagnosis_texts):
        diagnosis_index.append({
            "id": f"node:{node['node_id']}",
            "node_id": node["node_id"],
            "name": node["name"],
            "normalized_name": node["normalized_name"],
            "label": node["label"],
            "label_group": node["label_group"],
            "description": node["description"],
            "embedding_text": text,
        })

    relationship_lexical = build_lexical_metadata(relationship_texts)
    diagnosis_lexical = build_lexical_metadata(diagnosis_texts)

    relationship_lexical["tokenized_docs"] = [
        " ".join(tokens) for tokens in relationship_lexical["tokenized_docs"]
    ]
    diagnosis_lexical["tokenized_docs"] = [
        " ".join(tokens) for tokens in diagnosis_lexical["tokenized_docs"]
    ]

    save_json(output_dir / "relationship_index.json", relationship_index)
    save_json(output_dir / "diagnosis_index.json", diagnosis_index)

    np.save(output_dir / "relationship_embeddings.npy", relationship_embeddings)
    np.save(output_dir / "diagnosis_embeddings.npy", diagnosis_embeddings)

    save_json(output_dir / "relationship_lexical.json", relationship_lexical)
    save_json(output_dir / "diagnosis_lexical.json", diagnosis_lexical)

    metadata = {
        "model_name": model_name,
        "relationship_count": len(relationships),
        "diagnosis_count": len(diagnosis_nodes),
        "embedding_dimension": int(relationship_embeddings.shape[1]) if relationship_embeddings.ndim == 2 else 0,
        "normalized_embeddings": True,
        "diagnosis_label_group": DIAGNOSIS_LABEL_GROUP,
    }
    save_json(output_dir / "vector_search_metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="knowledge_graph.db")
    parser.add_argument("--output-dir", default="data/vector_search")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--debug-samples", type=int, default=3)
    parser.add_argument(
        "--debug-diagnosis-names",
        nargs="*",
        default=[],
        help="Optional exact diagnosis node names to print embedding text for.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        relationships = fetch_relationships(conn)
        relationship_sources = fetch_relationship_sources(conn)
        diagnosis_nodes = fetch_diagnosis_nodes(conn)
    finally:
        conn.close()

    by_source, by_target = build_relationship_lookup(relationships)

    relationship_texts = [relationship_to_text(rel) for rel in relationships]
    diagnosis_texts = [
        diagnosis_node_to_text(
            node,
            by_source.get(node["node_id"], []),
            by_target.get(node["node_id"], []),
        )
        for node in diagnosis_nodes
    ]

    print_sample_embeddings(
        relationships=relationships,
        relationship_texts=relationship_texts,
        diagnosis_nodes=diagnosis_nodes,
        diagnosis_texts=diagnosis_texts,
        num_samples=args.debug_samples,
    )

    print_specific_diagnosis_samples(
        diagnosis_nodes=diagnosis_nodes,
        diagnosis_texts=diagnosis_texts,
        names=args.debug_diagnosis_names,
    )

    relationship_embeddings = build_embeddings(
        relationship_texts,
        args.model,
        args.batch_size,
    )

    diagnosis_embeddings = build_embeddings(
        diagnosis_texts,
        args.model,
        args.batch_size,
    )

    save_outputs(
        output_dir=output_dir,
        relationships=relationships,
        relationship_texts=relationship_texts,
        relationship_embeddings=relationship_embeddings,
        relationship_sources=relationship_sources,
        diagnosis_nodes=diagnosis_nodes,
        diagnosis_texts=diagnosis_texts,
        diagnosis_embeddings=diagnosis_embeddings,
        model_name=args.model,
    )

    print(f"\nBuilt relationship + diagnosis indexes from database: {db_path.resolve()}")
    print(f"Relationships embedded: {len(relationships)}")
    print(f"Diagnosis nodes embedded: {len(diagnosis_nodes)}")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()