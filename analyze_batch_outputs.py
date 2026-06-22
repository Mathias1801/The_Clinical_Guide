import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "batch_outputs"

PROMPT_LABELS = {
    "kg_full": "With KG + full PSO note",
    "no_kg": "Without KG",
    "kg_weighted": "With KG weighted as main evidence",
    "kg_p_only": "With KG + P-section only",
    "kg_only": "KG only",
}

SHORT_PROMPT_LABELS = {
    "kg_full": "KG + PSO",
    "no_kg": "No KG",
    "kg_weighted": "KG weighted",
    "kg_p_only": "KG + P",
    "kg_only": "KG only",
}

JUDGE_CATEGORY_COLORS = {
    "retrieval": {
        "control_label_match": "#1f77b4",
        "included_nodes": "#6baed6",
    },
    "output": {
        "kg_influence": "#ff7f0e",
        "context_evaluator": "#fdae6b",
        "classification_match": "#fdbe85",
    },
    "kg_comparison": {
        "kg_improved_over_no_kg": "#d62728",
        "kg_visible_influence": "#ff9896",
    },
    "meta": {
        "meta_judge": "#2ca02c",
    },
    "arena": {
        "arena_winner": "#9467bd",
    },
}


st.set_page_config(page_title="Batch Output Analyzer", layout="wide")


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_files(output_dir):
    files = sorted(Path(output_dir).glob("*.json"))
    rows = []
    errors = []

    for path in files:
        if path.name.startswith("manifest_"):
            continue

        try:
            rows.append({
                "path": path,
                "file_name": path.name,
                "data": load_json_file(path),
            })
        except Exception as e:
            errors.append({"file": str(path), "error": str(e)})

    return rows, errors


def get_nested(data, path, default=None):
    current = data

    for key in path:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def pass_rate(series):
    valid = series.dropna()

    if len(valid) == 0:
        return 0.0

    return float((valid == True).mean())


def bool_label(value):
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "N/A"


def dataframe_from_records(records, fields):
    return pd.DataFrame([
        {field: item.get(field) for field in fields}
        for item in records or []
    ])


def extract_case_summary(item):
    data = item["data"]
    assessment = data.get("test_assessment_summary") or {}
    retrieval = assessment.get("retrieval_judges") or {}
    output_assessments = assessment.get("output_assessments") or {}
    arena = assessment.get("arena") or {}

    kg_comparison = (
        assessment.get("kg_comparison_judges")
        or get_nested(data, ["judges", "kg_comparison_judge_result"], {})
        or {}
    )
    kg_comparisons = kg_comparison.get("comparisons") or {}

    selected_nodes = get_nested(data, ["retrieval", "selected_node_ids"], []) or []
    selected_relations = get_nested(data, ["retrieval", "selected_relation_results"], []) or []
    expanded_relations = get_nested(data, ["retrieval", "expanded_relations"], []) or []

    winner_keys = arena.get("winner_keys") or []

    prompt_order = {
        "kg_full": 1,
        "kg_weighted": 2,
        "no_kg": 3,
        "kg_p_only": 4,
        "kg_only": 5,
    }

    winner_keys = sorted(
        winner_keys,
        key=lambda key: prompt_order.get(key, 999),
    )

    winner_labels = [
        SHORT_PROMPT_LABELS.get(key, PROMPT_LABELS.get(key, key))
        for key in winner_keys
    ]

    if not winner_labels:
        arena_winner = None
    elif len(winner_labels) == 1:
        arena_winner = winner_labels[0]
    else:
        arena_winner = "Tie: " + ", ".join(winner_labels)

    row = {
        "file_name": item["file_name"],
        "item_id": data.get("item_id"),
        "source_file": get_nested(data, ["source", "file_name"]),
        "row_number": get_nested(data, ["source", "row_number"]),
        "expected_diagnosis": get_nested(data, ["input", "diagnosis"]),
        "started_at": get_nested(data, ["timestamps", "started_at"]),
        "finished_at": get_nested(data, ["timestamps", "finished_at"]),
        "provider": get_nested(data, ["settings", "provider"]),
        "model_name": get_nested(data, ["settings", "model_name"]),
        "worker_base_url": get_nested(data, ["worker", "base_url"]),
        "skipped": data.get("skipped", False),
        "error_count": len(data.get("errors") or []),
        "retrieval_control_label_match": get_nested(retrieval, ["control_label_match", "accepted"]),
        "retrieval_included_nodes": get_nested(retrieval, ["included_nodes", "accepted"]),
        "arena_winner": arena_winner,
        "arena_is_tie": arena.get("is_tie") if arena else None,
        "arena_inconsistent_pair_count": arena.get("inconsistent_pair_count") if arena else None,
        "selected_node_count": len(selected_nodes),
        "selected_relation_count": len(selected_relations),
        "expanded_relation_count": len(expanded_relations),
    }

    for output_key, output_data in output_assessments.items():
        judges = output_data.get("output_judges") or {}
        meta = output_data.get("case_meta_judge") or {}

        row[f"{output_key}_kg_influence"] = get_nested(judges, ["kg_influence", "accepted"])
        row[f"{output_key}_context_evaluator"] = get_nested(judges, ["context_evaluator", "accepted"])
        row[f"{output_key}_classification_match"] = get_nested(judges, ["classification_match", "accepted"])
        row[f"{output_key}_meta_judge"] = get_nested(meta, ["meta_judge", "accepted"])

    for output_key, comparison_data in kg_comparisons.items():
        row[f"{output_key}_kg_improved_over_no_kg"] = get_nested(
            comparison_data,
            ["kg_improved_over_no_kg", "accepted"],
        )
        row[f"{output_key}_kg_visible_influence"] = get_nested(
            comparison_data,
            ["kg_visible_influence", "accepted"],
        )

    return row


def build_summary_dataframe(items):
    return pd.DataFrame([extract_case_summary(item) for item in items])


def get_output_keys(df):
    keys = set()

    for col in df.columns:
        for suffix in [
            "_kg_influence",
            "_context_evaluator",
            "_classification_match",
            "_meta_judge",
            "_kg_improved_over_no_kg",
            "_kg_visible_influence",
        ]:
            if col.endswith(suffix):
                keys.add(col.removesuffix(suffix))

    return sorted(keys)


def build_prompt_judge_long_dataframe(df):
    output_keys = get_output_keys(df)

    judge_specs = [
        {
            "judge": "kg_influence",
            "label": "KG influence (old)",
            "category": "Output judges",
            "color": JUDGE_CATEGORY_COLORS["output"]["kg_influence"],
        },
        {
            "judge": "context_evaluator",
            "label": "Context evaluator",
            "category": "Output judges",
            "color": JUDGE_CATEGORY_COLORS["output"]["context_evaluator"],
        },
        {
            "judge": "classification_match",
            "label": "Classification match",
            "category": "Output judges",
            "color": JUDGE_CATEGORY_COLORS["output"]["classification_match"],
        },
        {
            "judge": "meta_judge",
            "label": "Meta judge",
            "category": "Meta judges",
            "color": JUDGE_CATEGORY_COLORS["meta"]["meta_judge"],
        },
        {
            "judge": "kg_improved_over_no_kg",
            "label": "KG improved vs No-KG",
            "category": "KG comparison judges",
            "color": JUDGE_CATEGORY_COLORS["kg_comparison"]["kg_improved_over_no_kg"],
        },
        {
            "judge": "kg_visible_influence",
            "label": "KG visible influence",
            "category": "KG comparison judges",
            "color": JUDGE_CATEGORY_COLORS["kg_comparison"]["kg_visible_influence"],
        },
    ]

    rows = []

    for key in output_keys:
        for spec in judge_specs:
            col = f"{key}_{spec['judge']}"
            if col not in df.columns:
                continue

            cases_with_result = int(df[col].dropna().shape[0])

            if cases_with_result == 0:
                continue

            rows.append({
                "prompt_key": key,
                "prompt_label": PROMPT_LABELS.get(key, key),
                "short_prompt_label": SHORT_PROMPT_LABELS.get(key, key),
                "judge": spec["judge"],
                "judge_label": spec["label"],
                "category": spec["category"],
                "pass_rate": pass_rate(df[col]) * 100,
                "cases_with_result": cases_with_result,
                "passes": int((df[col] == True).sum()),
                "fails": int((df[col] == False).sum()),
                "color": spec["color"],
            })

    return pd.DataFrame(rows)


def build_kg_comparison_long_dataframe(df):
    output_keys = get_output_keys(df)

    judge_specs = [
        {
            "judge": "kg_improved_over_no_kg",
            "label": "KG improved vs No-KG",
            "color": JUDGE_CATEGORY_COLORS["kg_comparison"]["kg_improved_over_no_kg"],
        },
        {
            "judge": "kg_visible_influence",
            "label": "KG visible influence",
            "color": JUDGE_CATEGORY_COLORS["kg_comparison"]["kg_visible_influence"],
        },
    ]

    rows = []

    for key in output_keys:
        if key == "no_kg":
            continue

        for spec in judge_specs:
            col = f"{key}_{spec['judge']}"
            if col not in df.columns:
                continue

            cases_with_result = int(df[col].dropna().shape[0])

            if cases_with_result == 0:
                continue

            rows.append({
                "prompt_key": key,
                "prompt_label": PROMPT_LABELS.get(key, key),
                "short_prompt_label": SHORT_PROMPT_LABELS.get(key, key),
                "judge": spec["judge"],
                "judge_label": spec["label"],
                "pass_rate": pass_rate(df[col]) * 100,
                "cases_with_result": cases_with_result,
                "passes": int((df[col] == True).sum()),
                "fails": int((df[col] == False).sum()),
                "color": spec["color"],
            })

    return pd.DataFrame(rows)


def build_retrieval_judge_dataframe(df):
    rows = [
        {
            "judge": "control_label_match",
            "judge_label": "Control label match",
            "category": "Retrieval judges",
            "pass_rate": pass_rate(df["retrieval_control_label_match"]) * 100,
            "cases_with_result": int(df["retrieval_control_label_match"].dropna().shape[0]),
            "passes": int((df["retrieval_control_label_match"] == True).sum()),
            "fails": int((df["retrieval_control_label_match"] == False).sum()),
            "color": JUDGE_CATEGORY_COLORS["retrieval"]["control_label_match"],
        },
        {
            "judge": "included_nodes",
            "judge_label": "Included nodes",
            "category": "Retrieval judges",
            "pass_rate": pass_rate(df["retrieval_included_nodes"]) * 100,
            "cases_with_result": int(df["retrieval_included_nodes"].dropna().shape[0]),
            "passes": int((df["retrieval_included_nodes"] == True).sum()),
            "fails": int((df["retrieval_included_nodes"] == False).sum()),
            "color": JUDGE_CATEGORY_COLORS["retrieval"]["included_nodes"],
        },
    ]

    return pd.DataFrame(rows)


def add_bar_labels(ax, values, horizontal=False):
    if horizontal:
        for bar, value in zip(ax.patches, values):
            if pd.isna(value):
                continue

            width = bar.get_width()
            x = width - 3 if width >= 12 else width + 2
            ha = "right" if width >= 12 else "left"
            color = "white" if width >= 12 else "black"

            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.0f}",
                ha=ha,
                va="center",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
    else:
        for bar, value in zip(ax.patches, values):
            if pd.isna(value):
                continue

            height = bar.get_height()
            y = height - 4 if height >= 12 else height + 2
            va = "top" if height >= 12 else "bottom"
            color = "white" if height >= 12 else "black"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.0f}",
                ha="center",
                va=va,
                fontsize=8,
                color=color,
                fontweight="bold",
            )


def render_retrieval_judge_plot(df):
    retrieval_df = build_retrieval_judge_dataframe(df)

    st.markdown("### Retrieval judge verdicts")

    fig, ax = plt.subplots(figsize=(5.8, 2.4))

    values = retrieval_df["pass_rate"].tolist()
    colors = retrieval_df["color"].tolist()

    ax.barh(retrieval_df["judge_label"], values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pass rate (%)")
    ax.set_ylabel("")
    ax.set_title("Retrieval judge pass rates")

    add_bar_labels(ax, values, horizontal=True)

    fig.tight_layout()
    st.pyplot(fig)

    display_df = retrieval_df.copy()
    display_df["pass_rate"] = display_df["pass_rate"].round(1).astype(str) + "%"
    st.dataframe(
        display_df[["judge_label", "pass_rate", "passes", "fails", "cases_with_result"]],
        width="stretch",
        hide_index=True,
    )


def render_prompt_judge_plot(df):
    long_df = build_prompt_judge_long_dataframe(df)

    st.markdown("### Prompt-style judge verdicts")

    if long_df.empty:
        st.info("No prompt judge verdicts found.")
        return

    prompt_order = list(dict.fromkeys(long_df["short_prompt_label"].tolist()))
    judge_order = list(dict.fromkeys(long_df["judge_label"].tolist()))

    fig, ax = plt.subplots(figsize=(10.5, 4.6))

    bar_width = min(0.15, 0.8 / max(1, len(judge_order)))
    x_positions = range(len(prompt_order))

    for judge_idx, judge_label in enumerate(judge_order):
        subset = long_df[long_df["judge_label"] == judge_label]
        values = []

        for prompt_label in prompt_order:
            match = subset[subset["short_prompt_label"] == prompt_label]
            values.append(float(match["pass_rate"].iloc[0]) if not match.empty else 0.0)

        color = subset["color"].iloc[0] if not subset.empty else None
        offsets = [x + (judge_idx - (len(judge_order) - 1) / 2) * bar_width for x in x_positions]

        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=judge_label,
            color=color,
        )

        for bar, value in zip(bars, values):
            if value <= 0:
                continue

            y = value - 4 if value >= 12 else value + 2
            va = "top" if value >= 12 else "bottom"
            color_text = "white" if value >= 12 else "black"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.0f}",
                ha="center",
                va=va,
                fontsize=8,
                color=color_text,
                fontweight="bold",
            )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(prompt_order, rotation=0, ha="center")
    ax.set_ylabel("Pass rate (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_title("Each judge by prompt style")
    ax.legend(title="Judge", fontsize=8, title_fontsize=9)

    fig.tight_layout()
    st.pyplot(fig)

    display_df = long_df.copy()
    display_df["pass_rate"] = display_df["pass_rate"].round(1).astype(str) + "%"
    st.dataframe(
        display_df[
            [
                "prompt_label",
                "judge_label",
                "category",
                "pass_rate",
                "passes",
                "fails",
                "cases_with_result",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_kg_comparison_judge_plot(df):
    long_df = build_kg_comparison_long_dataframe(df)

    st.markdown("### KG comparison judge verdicts")

    if long_df.empty:
        st.info("No KG comparison judge verdicts found.")
        return

    prompt_order = list(dict.fromkeys(long_df["short_prompt_label"].tolist()))
    judge_order = list(dict.fromkeys(long_df["judge_label"].tolist()))

    fig, ax = plt.subplots(figsize=(8.5, 3.4))

    bar_width = 0.25
    x_positions = range(len(prompt_order))

    for judge_idx, judge_label in enumerate(judge_order):
        subset = long_df[long_df["judge_label"] == judge_label]
        values = []

        for prompt_label in prompt_order:
            match = subset[subset["short_prompt_label"] == prompt_label]
            values.append(float(match["pass_rate"].iloc[0]) if not match.empty else 0.0)

        color = subset["color"].iloc[0] if not subset.empty else None
        offsets = [x + (judge_idx - (len(judge_order) - 1) / 2) * bar_width for x in x_positions]

        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=judge_label,
            color=color,
        )

        for bar, value in zip(bars, values):
            if value <= 0:
                continue

            y = value - 4 if value >= 12 else value + 2
            va = "top" if value >= 12 else "bottom"
            color_text = "white" if value >= 12 else "black"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.0f}",
                ha="center",
                va=va,
                fontsize=8,
                color=color_text,
                fontweight="bold",
            )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(prompt_order, rotation=0, ha="center")
    ax.set_ylabel("Pass rate (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_title("KG comparison judges")
    ax.legend(title="Judge", fontsize=8, title_fontsize=9)

    fig.tight_layout()
    st.pyplot(fig)

    display_df = long_df.copy()
    display_df["pass_rate"] = display_df["pass_rate"].round(1).astype(str) + "%"
    st.dataframe(
        display_df[
            [
                "prompt_label",
                "judge_label",
                "pass_rate",
                "passes",
                "fails",
                "cases_with_result",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_arena_plot(df):
    st.markdown("### Arena winners")

    if "arena_winner" not in df.columns:
        st.info("No arena column found.")
        return

    arena_counts = (
        df["arena_winner"]
        .fillna("No arena result")
        .value_counts()
        .reset_index()
    )
    arena_counts.columns = ["arena_winner", "cases"]

    if arena_counts.empty:
        st.info("No arena results found.")
        return

    arena_counts = arena_counts.sort_values("cases", ascending=True)

    labels = arena_counts["arena_winner"].tolist()
    values = arena_counts["cases"].tolist()

    fig_height = max(2.8, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    bars = ax.barh(
        labels,
        values,
        color=JUDGE_CATEGORY_COLORS["arena"]["arena_winner"],
    )

    ax.set_xlabel("Cases")
    ax.set_ylabel("")
    ax.set_title("Final arena winners")

    max_value = max(values) if values else 1
    ax.set_xlim(0, max_value + 1)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        arena_counts.sort_values("cases", ascending=False),
        width="stretch",
        hide_index=True,
    )


def render_judge_verdict_charts(df):
    st.markdown("## Judge verdict comparison")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases", len(df))
    c2.metric("Cases with errors", int((df["error_count"] > 0).sum()) if not df.empty else 0)
    c3.metric("Avg selected nodes", round(df["selected_node_count"].mean(), 1) if not df.empty else 0)
    c4.metric("Avg selected relations", round(df["selected_relation_count"].mean(), 1) if not df.empty else 0)

    tab_retrieval, tab_prompt, tab_kg_comparison, tab_arena = st.tabs([
        "Retrieval judges",
        "Prompt judges",
        "KG comparison judges",
        "Arena",
    ])

    with tab_retrieval:
        render_retrieval_judge_plot(df)

    with tab_prompt:
        render_prompt_judge_plot(df)

    with tab_kg_comparison:
        render_kg_comparison_judge_plot(df)

    with tab_arena:
        render_arena_plot(df)


def render_filters(df):
    st.sidebar.header("Filters")

    query = st.sidebar.text_input("Search case", "")
    show_errors_only = st.sidebar.checkbox("Only cases with errors", value=False)

    winners = sorted([x for x in df["arena_winner"].dropna().unique()]) if "arena_winner" in df.columns else []
    selected_winners = st.sidebar.multiselect("Arena winner", winners)

    sources = sorted([x for x in df["source_file"].dropna().unique()])
    selected_sources = st.sidebar.multiselect("Source file", sources)

    filtered = df.copy()

    if query.strip():
        q = query.strip().lower()
        mask = (
            filtered["item_id"].fillna("").str.lower().str.contains(q)
            | filtered["file_name"].fillna("").str.lower().str.contains(q)
            | filtered["expected_diagnosis"].fillna("").str.lower().str.contains(q)
        )
        filtered = filtered[mask]

    if show_errors_only:
        filtered = filtered[filtered["error_count"] > 0]

    if selected_winners:
        filtered = filtered[filtered["arena_winner"].isin(selected_winners)]

    if selected_sources:
        filtered = filtered[filtered["source_file"].isin(selected_sources)]

    return filtered


def first_existing_col(df, cols):
    for col in cols:
        if col in df.columns:
            return col
    return None


def render_case_picker(df):
    st.markdown("## Case list")

    optional_cols = [
        "row_number",
        "expected_diagnosis",
        "arena_winner",
        "retrieval_control_label_match",
        "retrieval_included_nodes",
        "kg_full_kg_improved_over_no_kg",
        "kg_full_kg_visible_influence",
        "kg_weighted_kg_improved_over_no_kg",
        "kg_weighted_kg_visible_influence",
        "selected_node_count",
        "selected_relation_count",
        "error_count",
        "file_name",
    ]

    cols = [c for c in optional_cols if c in df.columns]

    st.dataframe(df[cols], width="stretch", hide_index=True)

    options = []

    for _, row in df.iterrows():
        label = (
            f"Row {row.get('row_number')} | "
            f"{row.get('expected_diagnosis')} | "
            f"winner={row.get('arena_winner') or 'N/A'} | "
            f"{row.get('file_name')}"
        )
        options.append((label, row.get("file_name")))

    if not options:
        return None

    selected_label = st.selectbox("Inspect case", [x[0] for x in options])
    return dict(options)[selected_label]


def render_input_step(data):
    st.markdown("## 1. Input row and label")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Row", get_nested(data, ["source", "row_number"], "N/A"))
    c2.metric("Expected diagnosis", get_nested(data, ["input", "diagnosis"], "N/A"))
    c3.metric("Source file", get_nested(data, ["source", "file_name"], "N/A"))
    c4.metric("Errors", len(data.get("errors") or []))

    st.markdown("### Patient note")
    st.code(get_nested(data, ["input", "pso_note"], ""), language="text")

    if data.get("errors"):
        with st.expander("Errors", expanded=True):
            for err in data.get("errors") or []:
                st.error(err)


def render_query_step(data):
    st.markdown("## 2. Improved query")

    query_processing = data.get("query_processing") or {}

    c1, c2 = st.columns(2)
    c1.metric("Structured query source", query_processing.get("structured_query_source", "N/A"))
    c2.metric("Enhanced query length", len(str(query_processing.get("enhanced_query", ""))))

    st.markdown("### Enhanced query")
    st.code(query_processing.get("enhanced_query", ""), language="text")

    with st.expander("Structured query JSON", expanded=False):
        st.json(query_processing.get("structured_query", {}))


def render_nodes_step(data):
    st.markdown("## 3. Fetched diagnosis nodes")

    selected_sets = get_nested(data, ["retrieval", "selected_diagnosis_result_sets"], {}) or {}
    reranked_sets = get_nested(data, ["retrieval", "reranked_diagnosis_result_sets"], {}) or {}
    result_sets = get_nested(data, ["retrieval", "diagnosis_result_sets"], {}) or {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Search methods", len(result_sets))
    c2.metric("Selected node IDs", len(get_nested(data, ["retrieval", "selected_node_ids"], []) or []))
    c3.metric("Total selected rows", sum(len(v or []) for v in selected_sets.values()))

    fields = [
        "node_id",
        "name",
        "label",
        "label_group",
        "llm_relevance",
        "score",
        "dense_score",
        "lexical_score",
        "matched_term_count",
        "matched_terms",
        "payload_reason",
        "description",
    ]

    tabs = st.tabs(["Selected nodes", "Reranked nodes", "Initial retrieval"])

    with tabs[0]:
        if selected_sets:
            for search_name, records in selected_sets.items():
                st.markdown(f"### {search_name}")
                st.dataframe(dataframe_from_records(records, fields), width="stretch", hide_index=True)
        else:
            st.info("No selected diagnosis nodes found.")

    with tabs[1]:
        if reranked_sets:
            for search_name, records in reranked_sets.items():
                st.markdown(f"### {search_name}")
                st.dataframe(dataframe_from_records(records, fields), width="stretch", hide_index=True)
        else:
            st.info("No reranked diagnosis nodes found.")

    with tabs[2]:
        if result_sets:
            for search_name, records in result_sets.items():
                st.markdown(f"### {search_name}")
                st.dataframe(dataframe_from_records(records, fields), width="stretch", hide_index=True)
        else:
            st.info("No initial diagnosis retrieval found.")


def render_relations_step(data):
    st.markdown("## 4. Fetched relations")

    selected_relations = get_nested(data, ["retrieval", "selected_relation_results"], []) or []
    reranked_relations = get_nested(data, ["retrieval", "reranked_relation_results"], []) or []
    expanded_relations = get_nested(data, ["retrieval", "expanded_relations"], []) or []

    c1, c2, c3 = st.columns(3)
    c1.metric("Expanded relations", len(expanded_relations))
    c2.metric("Reranked relations", len(reranked_relations))
    c3.metric("Selected relations", len(selected_relations))

    fields = [
        "relationship_id",
        "hop_depth",
        "hop_weight",
        "source_name",
        "type",
        "target_name",
        "llm_relevance",
        "score",
        "original_score",
        "payload_reason",
        "description",
    ]

    tabs = st.tabs(["Selected", "Reranked", "Expanded"])

    with tabs[0]:
        st.dataframe(dataframe_from_records(selected_relations, fields), width="stretch", hide_index=True)

    with tabs[1]:
        st.dataframe(dataframe_from_records(reranked_relations, fields), width="stretch", hide_index=True)

    with tabs[2]:
        st.dataframe(dataframe_from_records(expanded_relations, fields), width="stretch", hide_index=True)


def render_payload_step(data):
    st.markdown("## 5. Payload")

    payload = data.get("payload") or {}
    payload_relations = payload.get("selected_relations") or []

    st.metric("Payload triplets", len(payload_relations))

    fields = [
        "relationship_id",
        "source_name",
        "type",
        "target_name",
        "relevance_score",
        "vector_score",
        "description",
    ]

    st.dataframe(dataframe_from_records(payload_relations, fields), width="stretch", hide_index=True)

    with st.expander("Raw payload", expanded=False):
        st.json(payload)


def flatten_retrieval_judges(data):
    retrieval = get_nested(data, ["test_assessment_summary", "retrieval_judges"], {}) or {}
    rows = []

    for judge_name, judge_data in retrieval.items():
        if isinstance(judge_data, dict):
            rows.append({
                "Judge": judge_name,
                "Accepted": judge_data.get("accepted"),
                "Verdict": bool_label(judge_data.get("accepted")),
                "Score": judge_data.get("score"),
                "Reasoning": judge_data.get("reasoning"),
            })

    return pd.DataFrame(rows)


def render_payload_judge_step(data):
    st.markdown("## 6. Judge of payload")

    df = flatten_retrieval_judges(data)

    if df.empty:
        st.info("No retrieval/payload judge results found.")
        return

    st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("Payload judge reasoning", expanded=False):
        for _, row in df.iterrows():
            st.markdown(f"### {row['Judge']} — {row['Verdict']}")
            st.write(row["Reasoning"])


def render_prompt_output_step(data):
    st.markdown("## 7. Prompt outputs")

    prompts = get_nested(data, ["prompts", "diagnosis_prompts"], {}) or {}
    outputs = get_nested(data, ["outputs", "diagnosis_outputs"], {}) or {}

    output_keys = list(outputs.keys()) or list(prompts.keys())

    if not output_keys:
        st.info("No prompt outputs found.")
        return

    tabs = st.tabs([PROMPT_LABELS.get(key, key) for key in output_keys])

    for tab, key in zip(tabs, output_keys):
        with tab:
            st.markdown(f"### {PROMPT_LABELS.get(key, key)}")

            if key in prompts:
                with st.expander("Prompt", expanded=False):
                    st.code(prompts[key], language="text")

            st.markdown("#### Output")

            if outputs.get(key):
                st.write(outputs[key])
            else:
                st.warning("No output generated.")


def flatten_output_judges(data):
    output_assessments = get_nested(data, ["test_assessment_summary", "output_assessments"], {}) or {}
    rows = []

    for output_key, output_data in output_assessments.items():
        label = output_data.get("label") or PROMPT_LABELS.get(output_key, output_key)

        for judge_name, judge_data in (output_data.get("output_judges") or {}).items():
            if isinstance(judge_data, dict):
                rows.append({
                    "Prompt style": label,
                    "Output key": output_key,
                    "Section": "Output judge",
                    "Judge": judge_name,
                    "Accepted": judge_data.get("accepted"),
                    "Verdict": bool_label(judge_data.get("accepted")),
                    "Score": judge_data.get("score"),
                    "Reasoning": judge_data.get("reasoning"),
                })

        for judge_name, judge_data in (output_data.get("case_meta_judge") or {}).items():
            if isinstance(judge_data, dict):
                rows.append({
                    "Prompt style": label,
                    "Output key": output_key,
                    "Section": "Meta judge",
                    "Judge": judge_name,
                    "Accepted": judge_data.get("accepted"),
                    "Verdict": bool_label(judge_data.get("accepted")),
                    "Score": judge_data.get("score"),
                    "Reasoning": judge_data.get("reasoning"),
                })

    return pd.DataFrame(rows)


def flatten_kg_comparison_judges(data):
    kg_comparison = (
        get_nested(data, ["test_assessment_summary", "kg_comparison_judges"], {})
        or get_nested(data, ["judges", "kg_comparison_judge_result"], {})
        or {}
    )
    comparisons = kg_comparison.get("comparisons") or {}

    rows = []

    for output_key, comparison_data in comparisons.items():
        label = PROMPT_LABELS.get(output_key, output_key)

        for judge_name in ["kg_improved_over_no_kg", "kg_visible_influence"]:
            judge_data = comparison_data.get(judge_name)

            if isinstance(judge_data, dict):
                rows.append({
                    "Prompt style": label,
                    "Output key": output_key,
                    "Section": "KG comparison judge",
                    "Judge": judge_name,
                    "Accepted": judge_data.get("accepted"),
                    "Verdict": bool_label(judge_data.get("accepted")),
                    "Score": judge_data.get("score"),
                    "Reasoning": judge_data.get("reasoning"),
                })

    return pd.DataFrame(rows)


def render_output_judge_step(data):
    st.markdown("## 8. Judge of outputs")

    if data.get("judge_text_overview"):
        with st.expander("Simple judge overview text", expanded=False):
            st.text_area("Judge overview", data["judge_text_overview"], height=300)

    df = flatten_output_judges(data)
    kg_df = flatten_kg_comparison_judges(data)

    tabs = st.tabs(["Output and meta judges", "KG comparison judges"])

    with tabs[0]:
        if df.empty:
            st.info("No output judge results found.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)

            with st.expander("Output judge reasoning", expanded=False):
                for _, row in df.iterrows():
                    st.markdown(
                        f"### {row['Prompt style']} | {row['Section']} | {row['Judge']} — {row['Verdict']}"
                    )
                    st.write(row["Reasoning"])

    with tabs[1]:
        if kg_df.empty:
            st.info("No KG comparison judge results found.")
        else:
            st.dataframe(kg_df, width="stretch", hide_index=True)

            with st.expander("KG comparison judge reasoning", expanded=True):
                for _, row in kg_df.iterrows():
                    st.markdown(
                        f"### {row['Prompt style']} | {row['Judge']} — {row['Verdict']}"
                    )
                    st.write(row["Reasoning"])


def render_arena_step(data):
    st.markdown("## 9. Arena score")

    arena = get_nested(data, ["test_assessment_summary", "arena"], {}) or {}

    if not arena:
        st.info("No arena result found.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Winner", ", ".join(arena.get("winner_labels") or []) or "N/A")
    c2.metric("Tie", "Yes" if arena.get("is_tie") else "No")
    c3.metric("Inconsistent pairs", arena.get("inconsistent_pair_count", 0))

    ranking = arena.get("ranking") or []
    rows = []

    for idx, item in enumerate(ranking, start=1):
        score = item.get("score") or {}
        rows.append({
            "Rank": idx,
            "Prompt style": item.get("label"),
            "Output key": item.get("key"),
            "Wins": score.get("wins"),
            "Losses": score.get("losses"),
            "Points": score.get("points"),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Raw arena summary", expanded=False):
        st.json(arena)


def render_raw_json(data):
    st.markdown("## Raw JSON")
    st.json(data)


def main():
    st.title("Batch Output Analyzer")

    data_dir = PROJECT_ROOT / "data"

    output_dirs = sorted([
        path for path in data_dir.glob("batch_outputs*")
        if path.is_dir()
    ])

    if not output_dirs:
        st.warning("No batch output folders found.")
        st.stop()

    selected_output_dir = st.sidebar.selectbox(
        "Batch output folder",
        output_dirs,
        index=output_dirs.index(DEFAULT_OUTPUT_DIR) if DEFAULT_OUTPUT_DIR in output_dirs else 0,
        format_func=lambda path: path.name,
    )

    items, load_errors = load_json_files(selected_output_dir)

    if load_errors:
        with st.sidebar.expander("Load errors", expanded=False):
            st.json(load_errors)

    if not items:
        st.warning("No batch JSON files found.")
        st.stop()

    df = build_summary_dataframe(items)
    filtered_df = render_filters(df)

    render_judge_verdict_charts(filtered_df)

    st.markdown("---")

    selected_file_name = render_case_picker(filtered_df)

    if not selected_file_name:
        st.stop()

    selected_item = next(item for item in items if item["file_name"] == selected_file_name)
    data = selected_item["data"]

    st.markdown("---")
    st.markdown("# Chronological case inspection")

    inspection_tabs = st.tabs([
        "1 Input",
        "2 Improved query",
        "3 Nodes",
        "4 Relations",
        "5 Payload",
        "6 Payload judge",
        "7 Outputs",
        "8 Output judges",
        "9 Arena",
        "Raw JSON",
    ])

    with inspection_tabs[0]:
        render_input_step(data)

    with inspection_tabs[1]:
        render_query_step(data)

    with inspection_tabs[2]:
        render_nodes_step(data)

    with inspection_tabs[3]:
        render_relations_step(data)

    with inspection_tabs[4]:
        render_payload_step(data)

    with inspection_tabs[5]:
        render_payload_judge_step(data)

    with inspection_tabs[6]:
        render_prompt_output_step(data)

    with inspection_tabs[7]:
        render_output_judge_step(data)

    with inspection_tabs[8]:
        render_arena_step(data)

    with inspection_tabs[9]:
        render_raw_json(data)


if __name__ == "__main__":
    main()