from textwrap import dedent
from typing import Optional
import json

import instructor
from judges.base import Judgment


JUDGE_GENERAL_RULES = """
General judging rules:
- Evaluate substance, not style.
- Ignore response length unless missingness or harmful verbosity directly affects usefulness.
- Ignore formatting quality, fluency, or authoritative tone.
- Do not reward confident wording unless it is supported by the context.
- Base the decision only on the provided context, retrieved information, and optional reference.
""".strip()


OUTPUT_RULES_BOOL = """
Output format:
- REASONING: short reasoning paragraph.
- SCORE: True or False.
- SCORE must agree with the judge-specific decision rules above.
""".strip()

CONTROL_LABEL_RULES = """
Control label rules:
- The control label is optional reference information about the expected diagnosis.
- Use it only as a reference point for judging clinical alignment.
- Do not require exact wording unless the specific judge is explicitly a label-match/classification judge.
- Do not automatically fail an output just because the control label is absent.
- Do not automatically pass an output just because the control label is mentioned.
- Prefer outputs that are clinically grounded, context-supported, and compatible with the control label.
- Penalize outputs that contradict the control label when the clinical context supports the label.
- Penalize outputs that mention the control label but reason poorly, unsafely, or without support.
""".strip()

def make_ollama_v1_url(base_url):
    if not base_url:
        return None

    clean = str(base_url).rstrip("/")

    if clean.endswith("/v1"):
        return clean

    return f"{clean}/v1"


class RemoteBaseJudge:
    def __init__(self, model: str, base_url: str = None):
        self.model = model
        self.base_url = make_ollama_v1_url(base_url)

    def _build_messages(self, user_prompt: str, system_prompt: Optional[str] = None):
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": user_prompt,
        })

        return messages

    def _judge(self, user_prompt: str, system_prompt: Optional[str] = None):
        messages = self._build_messages(user_prompt, system_prompt)

        provider_kwargs = {}

        if self.base_url:
            provider_kwargs["base_url"] = self.base_url

        client = instructor.from_provider(
            self.model,
            **provider_kwargs,
        )

        judgment = client.chat.completions.create(
            messages=messages,
            temperature=0.0,
            response_model=Judgment,
        )

        return judgment.reasoning, judgment.score

def build_retrieved_relations_text(selected_relations):
    if not selected_relations:
        return "No information was retrieved."

    lines = []
    for rel in selected_relations:
        line = f"- {rel['source_name']} --{rel['type']}--> {rel['target_name']}"
        if rel.get("description"):
            line += f" | {rel['description']}"
        lines.append(line)

    return "\n".join(lines)


class RetrievalCoverageJudge(RemoteBaseJudge):
    def judge(self, input: str, output: str = None, expected: str = None) -> Judgment:
        system_prompt = (
            "You are a clinical label-match judge. "
            "Your only task is to decide whether the retrieved knowledge graph information explicitly contains "
            "the control diagnosis label or a very close unambiguous synonym of that exact label. "
            "You are NOT judging general retrieval quality, usefulness, plausibility, or broader clinical relevance. "
            "You must behave like a binary classifier. "
            f"{JUDGE_GENERAL_RULES}"
        )

        user_prompt = dedent(
            f"""
            Task:
            Decide whether the retrieved knowledge graph information explicitly contains the control label
            or a very close unambiguous synonym of that exact label.

            Patient note:
            {input}

            Retrieved knowledge graph information:
            {output}

            Control label:
            {expected}

            Hard decision rules:
            Return True if:
            - the control label itself appears explicitly in the retrieved information, or
            - a clinically equivalent diagnosis/condition name appears explicitly, even if the wording differs, or
            - the retrieved label is a spelling, inflectional, Danish/English, abbreviation, or near-synonym variant of the same clinical concept, or
            - the retrieved label refers to the same base disease concept but omits generic suffixes such as "sygdom", "tilstand", "lidelse", or similar.

            Return False if:
            - no retrieved diagnosis/condition label refers to the same clinical concept as the control label
            - the match is only a broad parent category, related symptom, complication, risk factor, test, medication, or differential diagnosis

            Important constraints:
            - This is NOT a retrieval-quality judge
            - This is NOT a relevance judge
            - This is NOT a plausibility judge

            Required reasoning format:
            1. State the control label.
            2. Quote the exact matching retrieved label or a very close unambiguous synonym if one exists.
            3. The final boolean MUST agree with the reasoning. State True if a matching retrieved label or a very close unambiguous synonym is found.

            {OUTPUT_RULES_BOOL}
            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score, score_type="boolean")


class IncludedNodesJudge(RemoteBaseJudge):
    def judge(self, input: str, output: str = None) -> Judgment:
        system_prompt = (
            "You are a clinical relevancy judge. "
            "Your task is to identify whether included information in the retrieved search information is out of relevant scope. "
            f"{JUDGE_GENERAL_RULES}"
        )

        user_prompt = dedent(
            f"""
            Task:
            Decide whether information presented can be categorized as relevant clinical information when comparing the retrieved information to the user input text.

            User input:
            {input}

            Retrieved information:
            {output}

            Decision rubric:
            Return True if:
            - the retrieved information is clinically reasonable or relevant to the case at hand
            - extra information is present but does not dominate or seriously mislead the case

            Return False if:
            - most retrieved information is unrelated to the user input
            - the retrieved information is dominated by irrelevant or misleading nodes
            - the retrieved information would likely distract from or distort the clinical interpretation

            Important constraints:
            - You are NOT judging whether retrieval is complete.
            - You are NOT judging whether important information is missing.
            - You are NOT judging whether the exact control diagnosis is present.
            - You ONLY judge whether the included retrieved information is clinically reasonable for the case.
            - Do NOT assess coverage of the retrieved information.
            - Do assess relevancy of the retrieved information.

            Examples:

            Example 1:

            Patient note:
            "P: Dyspnø

            S: Pt. oplevede dyspnø og svimmelhed efter lang cykeltur i weekenden. Svært ved at få vejret efterfølgende. Mild brystsmerte som forsvandt efter ca. 1 time. Ingen feber eller andre symptomer nu. Ingen syge familiemedlemmer. Tager Imitrex ved behov for migræne som er under kontrol. Undgår stærkt lys. Tager Protonix for reflux og har det godt med det. Haft kataraktoperation for 4,5 mdr. siden. Syn ok siden da.

            O: Hals smidig. Ingen jugular venestase. Resp. let ekspiratorisk hvæsen bilat. Cor regelmæssig rytme uden mislyde. Muskuloskeletal let ødem i bilat. UE. Lungefunktionsundersøgelse normal. Rtg. thorax u.a. CBC normal."

            Relevant knowledge graph triplets:
            Dyspnø --KAN_MANIFESTERE_SIG_SOM--> Åndenød | Dyspnø er åndenød.
            Dyspnø --ER_ASSOCIERET_MED--> Svimmelhed | Dyspnø kan være associeret med svimmelhed.
            Brystsmerte --KAN_MANIFESTERE_SIG_SOM--> Kardiel smerte | Brystsmerte kan være kardiel.
            Hjertesvigt --HAR_SYMPTOM--> Dyspnø | Hjertesvigt kan give dyspnø.
            Hjertesvigt --HAR_KLINISK_FUND--> Pittingødem | Hjertesvigt kan give ødem.
            Astma --HAR_SYMPTOM--> Hvæsen | Astma kan give hvæsen.
            Obstruktiv lungesygdom --HAR_KLINISK_FUND--> Hvæsen | Obstruktiv lungesygdom kan give hvæsen.
            Lungeemboli --HAR_SYMPTOM--> Dyspnø | Lungeemboli kan give dyspnø.
            Kardiovaskulær sygdom --HAR_SYMPTOM--> Brystsmerte | Kardiovaskulær sygdom kan give brystsmerte.
            Gastroøsofageal reflukssygdom --KAN_MANIFESTERE_SIG_SOM--> Brystsmerte | Refluks kan give brystsmerter.
            Migræne --BEHANDLES_MED--> Triptan | Migræne behandles med triptaner.

            Judgment:

            The retrieved knowledge graph captures the core clinical problem of dyspnø with associated symptoms (svimmelhed, brystsmerte) and relevant differential diagnoses including cardiopulmonary causes such as hjertesvigt, astma og lungeemboli. Some additional triplets (e.g. migræne, refluks) reflect information present in the history and are clinically explainable, even if not central. The overall inclusion of nodes makes sense and preserves the primary clinical reasoning.

            True

            Example 2:

            Patient note:
            "P: Opfølgning efter akutbesøg

            S: Følte mig svimmel under gåtur i går. Kæreste greb mig før jeg faldt. BP næsten 200 på skadestuen. Hovedpine. BP normalt men stiger ca. en uge om måneden pga. rejser for arbejde. Spiser ikke godt og måler ikke BP under rejser. Tager lisinopril som foreskrevet. Depression går godt. Startede terapi sidste år. Går en gang om ugen. Godt støttesystem med kæreste, mor og bror. Lidt næsekatar fra sæsonbetingede allergier. Ingen brystsmerter eller åndenød.

            O: Ingen carotis mislyde. Lunger auskulteres klare bilat. Ingen hvæsen rallelyde eller rhonchi. Let 2/6 systolisk ejektionsmislyd. Let pittingødem i bilat. UE. BT forhøjet. EKG stabil ift. sidste år. Ekkokardiogram viser nedsat EF. Stabil hjertemislyd."

            Relevant knowledge graph triplets:
            Hypertension --HAR_SYMPTOM--> Hovedpine | Hypertension kan give hovedpine.
            Hypertension --HAR_SYMPTOM--> Svimmelhed | Hypertension kan give svimmelhed.
            Hypertension --KAN_MEDFØRE--> Hjertesvigt | Hypertension kan føre til hjertesvigt.
            Hjertesvigt --HAR_KLINISK_FUND--> Nedsat ejektionsfraktion | Hjertesvigt er associeret med nedsat ejektionsfraktion.
            Hjertesvigt --HAR_KLINISK_FUND--> Pittingødem | Hjertesvigt kan give pittingødem.
            Hjertesvigt --HAR_KLINISK_FUND--> Systolisk mislyd | Hjertesvigt kan være associeret med systolisk mislyd.
            Depression --BEHANDLES_MED--> Psykoterapi | Depression behandles med psykoterapi.
            Sæsonbetinget allergi --HAR_SYMPTOM--> Næsekatar | Sæsonbetinget allergi giver næsekatar.
            Influenza --HAR_SYMPTOM--> Feber | Influenza kan give feber.
            Hyperthyreose --HAR_SYMPTOM--> Vægttab | Hyperthyreose kan give vægttab.
            Osteoporose --KAN_MEDFØRE--> Fraktur | Osteoporose kan føre til frakturer.

            Judgment:

            The knowledge graph includes several correct and clinically relevant relationships related to hypertension and possible cardiac involvement. However, it also contains multiple unrelated and unjustified triplets (e.g. influenza, hyperthyreose, osteoporose) that do not align with the patient’s presentation. The amount of noise dilutes the clinical coherence, and it does not fully make sense why these nodes were included. The retrieval therefore fails to represent a focused and clinically meaningful interpretation.

            False
            
            Example 3:

            Patient note: 
            "P: Opfølgning på unormale blodprøver

            S: Haft rutineblodprøver sidste uge. Fik at vide blodsukker var højt. Ingen symptomer. Kost har været god. Undgår sukker. Holder øje med vægt. Powerwalker 30 min dagligt. Måler blodsukker hver morgen. Tager metformin 1000 mg dagligt. Blodtryk stabilt med lisinopril. Har blodtryksapparat hjemme. Ingen problemer med højre knæ efter ACL-operation for 5 år siden. Ingen brystsmerter, åndenød, opkastning, diarré, hovedpine eller mavesmerter. Ingen problemer med vandladning.

            O: Normocephal og atraumatisk. Hals smidig uden thyromegali eller lymfadenopati. Ingen carotis mislyde. Lunger auskulteres klare bilat. Ingen hvæsen eller rallelyde. Cor med 3/6 systolisk uddrivningsmislyd. Abdomen blød og uøm. Ingen underekstremitetsødem. HbA1c forhøjet til 8,1."

            Relevant knowledge graph triplets:
            Type 2-diabetes --HAR_KLINISK_FUND--> Forhøjet HbA1c | Type 2-diabetes er associeret med forhøjet HbA1c.
            Type 2-diabetes --BEHANDLES_MED--> Metformin | Type 2-diabetes behandles med metformin.
            Type 2-diabetes --ER_ASSOCIERET_MED--> Hyperglykæmi | Type 2-diabetes er associeret med hyperglykæmi.
            Hyperglykæmi --HAR_KLINISK_FUND--> Forhøjet blodsukker | Hyperglykæmi er karakteriseret ved forhøjet blodsukker.
            Hypertension --BEHANDLES_MED--> ACE-hæmmer | Hypertension behandles med ACE-hæmmere.
            Type 2-diabetes --HAR_RISIKOFAKTOR--> Usund kost | Usund kost øger risikoen for type 2-diabetes.
            Type 2-diabetes --HAR_RISIKOFAKTOR--> Fysisk inaktivitet | Fysisk inaktivitet øger risikoen for type 2-diabetes.
            Diabetes --KAN_MEDFØRE--> Mikrovaskulær komplikation | Diabetes kan føre til mikrovaskulære komplikationer.
            Mikrovaskulær komplikation --ER_EN_TYPE_AF--> Komplikation | Mikrovaskulær komplikation er en type komplikation.
            Hjerteklapsygdom --HAR_KLINISK_FUND--> Systolisk mislyd | Hjerteklapsygdom kan give systolisk mislyd.
            Kardiovaskulær sygdom --ER_ASSOCIERET_MED--> Diabetes | Diabetes er associeret med øget risiko for kardiovaskulær sygdom.

            Judgment:

            The retrieved knowledge graph captures the primary clinical issue of diabetes with elevated HbA1c and appropriate treatment. Additional nodes such as cardiovascular associations and murmur-related conditions are clinically plausible given the findings and context. While some triplets extend beyond the immediate problem, they remain medically relevant and interpretable. The inclusion therefore makes sense overall, and the core clinical signal is preserved.

            True
            {OUTPUT_RULES_BOOL}

            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score, score_type="boolean")
    

class ClassificationJudge(RemoteBaseJudge):
    def judge(self, input: str, output: str = None, expected: str = None) -> Judgment:
        system_prompt = (
            "You are a label-oriented clinical classification judge. "
            "Assess whether the output explicitly supports or states the expected diagnosis label. "
            f"{JUDGE_GENERAL_RULES}"
        )

        user_prompt = dedent(
            f"""
            Task:
            Decide whether the output matches the expected diagnosis label.

            Context:
            {input}

            Output to evaluate:
            {output}

            Expected label:
            {expected}

            Decision rubric:
            Return True only if:
            - the expected label is directly stated in the output, or
            - the output clearly identifies the same diagnosis using a close or unambiguous equivalent clinical term

            Return False if:
            - the expected label is absent
            - the output points to a different diagnosis
            - the match depends on weak inference or vague similarity
            - the output is too ambiguous to confirm the expected label

            Important:
            - Focus only on whether the expected label is present or clearly matched
            - Do not reward partially related diagnoses
            - Do not infer a match unless it is clinically clear

            {OUTPUT_RULES_BOOL}
            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score, score_type="boolean")

class ContextEvaluatorJudge(RemoteBaseJudge):
    def judge(self, input: str, output: str = None, expected: str = None) -> Judgment:
        system_prompt = (
            "You are a clinical accuracy and context-consistency judge. "
            "Use a strict decision tree to decide whether the output is supported by the context and is not wrong. "
            f"{JUDGE_GENERAL_RULES}"
        )

        user_prompt = dedent(
            f"""
            Task:
            Decide whether the output is contextually correct and not clinically misleading.

            Context:
            {input}

            Output to evaluate:
            {output}

            Optional control label:
            {expected or ""}

            {CONTROL_LABEL_RULES}

            Decision tree:
            Step 1: Direct contradiction check
            - Does the output contradict the provided context information?
            - If YES -> Return False
            - If NO -> Continue

            Step 2: Unsupported clinically meaningful claim check
            - Does the output add any clinically meaningful claim, implication, diagnosis, recommendation,
              risk statement, causal interpretation, or factual detail that cannot be supported?
            - If YES -> Return False
            - If NO -> Continue

            Step 3: Certainty escalation check
            - Does the output express stronger certainty than the context supports?
            - If YES -> Return False
            - If NO -> Continue

            Step 4: Misleading framing check
            - Even if not directly contradicted, could the output reasonably mislead a clinical reader
              because of omission, overgeneralization, or inaccurate emphasis?
            - If YES -> Return False
            - If NO -> Continue

            Step 5: Context consistency check
            - Is the output overall consistent with the context?
            - If NO -> Return False
            - If YES -> Continue

            Step 6: Final decision
            - Return True only if all prior checks passed

            {OUTPUT_RULES_BOOL}
            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score, score_type="boolean")
    

class KGImprovementOverNoKGJudge(RemoteBaseJudge):
    def judge(
        self,
        input: str,
        kg_output: str = None,
        no_kg_output: str = None,
        expected: str = None,
        kg_output_judges=None,
        no_kg_output_judges=None,
    ) -> Judgment:
        system_prompt = (
            "You are a clinical comparative evaluation judge. "
            "Your task is to decide whether a Knowledge Graph (KG) Retrieval Augmented Generation (RAG) KG/RAG-assisted output is clinically better than a no-KG output. "
            "Evaluate substance, not style. "
            "Use the patient note and control label as the main basis for the decision. "
            "Use prior judge results as supporting evidence, especially context-consistency and classification results, "
            "but do not blindly copy them. "
            f"{JUDGE_GENERAL_RULES}"
        )

        user_prompt = dedent(
            f"""
            Task:
            Compare the KG/RAG-assisted output against the no-KG output.

            Decide whether the KG/RAG-assisted output is clinically better than the no-KG output.

            Clinical context:
            {input}

            Control label:
            {expected or ""}

            No-KG output:
            {no_kg_output or ""}

            Prior judge results for no-KG output:
            {json.dumps(no_kg_output_judges or {}, ensure_ascii=False, indent=2)}

            KG/RAG-assisted output:
            {kg_output or ""}

            Prior judge results for KG/RAG-assisted output:
            {json.dumps(kg_output_judges or {}, ensure_ascii=False, indent=2)}

            Decision rubric:
            Return True if the KG/RAG-assisted output is better overall because it:
            - gives a more clinically correct or precise diagnosis
            - is better aligned with the patient note and control label
            - gives better-supported reasoning
            - identifies better differentials or next checks
            - avoids unsupported or misleading claims better than the no-KG output
            - uses KG information to improve clinical reasoning without adding harmful noise

            Return False if:
            - the no-KG output is better
            - both outputs are essentially equal
            - the KG/RAG-assisted output adds noise, irrelevant differentials, unsupported claims, or misleading emphasis
            - the KG/RAG-assisted output appears more KG-influenced but is not clinically better
            - the KG/RAG-assisted output is only stylistically different, not substantively better

            Important:
            - Do not reward the KG output merely because it used KG information.
            - Do not reward longer outputs.
            - Do not require exact wording of the control label if the clinical diagnosis is equivalent.
            - If both outputs are similarly correct and useful, return False.
            - The question is improvement, not merely influence.
            - The SCORE must be True only when KG clearly improved the output compared with no-KG.

            {OUTPUT_RULES_BOOL}
            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        return Judgment(reasoning=reasoning, score=score, score_type="boolean")


class KGVisibleInfluenceJudge(RemoteBaseJudge):
    def judge(
        self,
        input: str,
        kg_output: str = None,
        no_kg_output: str = None,
        expected: str = None,
    ) -> Judgment:
        system_prompt = (
            "You are a Knowledge Graph (KG) Retrieval Augmented Generation (RAG) KG-RAG influence visibility judge. "
            "Your task is to decide whether the retrieved KG/RAG evidence visibly influenced the KG-assisted output. "
            "You are not judging whether the KG output is clinically better. "
            f"{JUDGE_GENERAL_RULES}"
        )

        user_prompt = dedent(
            f"""
            Task:
            Decide whether the KG/RAG evidence visibly influenced the KG-assisted output.

            Clinical context:
            {input}

            Control label:
            {expected or ""}

            No-KG output:
            {no_kg_output or ""}

            KG/RAG-assisted output:
            {kg_output or ""}

            Core question:
            Can you see that the KG/RAG information mattered for the KG-assisted answer?

            Return True if:
            - the KG-assisted output includes reasoning, differentials, mechanisms, tests, symptoms, treatments, or follow-up considerations that are clearly traceable to the retrieved KG context
            - the KG-assisted output differs from the no-KG output in a way that reflects KG evidence
            - the KG-assisted output prioritizes, supports, or explains a diagnosis using concepts from the KG
            - the KG-assisted output includes KG-derived clinical relationships, even if it does not explicitly say "knowledge graph"

            Return False if:
            - the KG-assisted output could have been written almost identically without the KG
            - the KG-assisted output only repeats the patient note and does not visibly use retrieved KG information
            - the KG-assisted output mentions generic concepts that are not specifically traceable to the retrieved KG
            - the KG influence is absent, superficial, or impossible to distinguish from ordinary clinical reasoning

            Important:
            - This judge does NOT decide whether the KG output is better.
            - KG influence can be visible even if it made the output worse.
            - KG influence can be visible even if the final diagnosis is the same.
            - Focus only on whether the RAG component visibly affected the answer.
            - The SCORE must be True only when KG influence is visible in the KG-assisted output.

            {OUTPUT_RULES_BOOL}
            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        return Judgment(reasoning=reasoning, score=score, score_type="boolean")


def _normalize_judgment(judgment):
    raw_score = getattr(judgment, "score", False)

    if isinstance(raw_score, bool):
        accepted = raw_score
    elif isinstance(raw_score, str):
        accepted = raw_score.strip().lower() == "true"
    else:
        accepted = bool(raw_score)

    return {
        "accepted": accepted,
        "score": raw_score,
        "reasoning": getattr(judgment, "reasoning", ""),
    }

def build_case_judgment_summary(retrieval_results=None, output_results=None):
    lines = []

    if retrieval_results:
        lines.append("Retrieval judges:")
        for judge_name, result in retrieval_results.items():
            if judge_name in {
                "retrieved_text",
                "control_label_match_text",
                "model",
                "control_label",
            }:
                continue
            if not isinstance(result, dict):
                continue
            lines.append(
                f"{judge_name}: accepted={result['accepted']} | score={result['score']} | reasoning={result['reasoning']}"
            )

    if output_results:
        if lines:
            lines.append("")
        lines.append("Output judges:")
        for judge_name, result in output_results.items():
            if judge_name in {
                "meta_judge",
                "subjudge_summary",
                "model",
                "control_label",
                "kg_influence_included",
            }:
                continue
            if not isinstance(result, dict):
                continue
            lines.append(
                f"{judge_name}: accepted={result['accepted']} | score={result['score']} | reasoning={result['reasoning']}"
            )

    return "\n".join(lines)

def run_retrieval_judges(
    input_text,
    selected_relations,
    model_name,
    control_label=None,
    all_node_candidates_text=None,
    base_url=None,
):
    retrieved_text = build_retrieved_relations_text(selected_relations)
    clean_control_label = (control_label or "").strip()

    judges = {
        "control_label_match": RetrievalCoverageJudge(
            model=model_name,
            base_url=base_url,
        ),
        "included_nodes": IncludedNodesJudge(
            model=model_name,
            base_url=base_url,
        ),
    }

    results = {}

    coverage_input_text = (
        all_node_candidates_text
        if all_node_candidates_text
        else retrieved_text
    )

    results["control_label_match"] = _normalize_judgment(
        judges["control_label_match"].judge(
            input=input_text,
            output=coverage_input_text,
            expected=clean_control_label,
        )
    )

    results["included_nodes"] = _normalize_judgment(
        judges["included_nodes"].judge(
            input=input_text,
            output=retrieved_text,
        )
    )

    results["retrieved_text"] = retrieved_text
    results["control_label_match_text"] = coverage_input_text
    results["control_label"] = clean_control_label
    results["model"] = model_name

    return results

class JudgeAuditMetaJudge(RemoteBaseJudge):
    def judge(self, input: str, output: str = None, expected: str = None) -> Judgment:
        system_prompt = (
            "You are a clinical judge-audit meta judge. "
            "Your task is to decide whether the previous judge results are logically sound, "
            "consistent, and supported by the clinical context and evaluated output. "
            f"{JUDGE_GENERAL_RULES}"
        )

        user_prompt = dedent(
            f"""
            Task:
            Audit the previous judge results for this clinical output.

            Clinical context:
            {input}

            Output that was judged:
            {output}

            Previous judge results:
            {expected}

            Decision rubric:
            Return True if:
            - the previous judge results are internally consistent
            - the judge reasoning supports the boolean decisions
            - the judgments are clinically reasonable given the context and output
            - no major clinical error or unsupported claim was missed

            Return False if:
            - one or more judge results contradict their own reasoning
            - important clinical problems were missed
            - the judge results are too shallow or unsupported
            - the overall evaluation is unreliable

            Important:
            - You are not re-judging from scratch unless needed.
            - Your job is to audit whether the judge process was sound.
            - Be strict when judge reasoning and verdicts do not align.

            {OUTPUT_RULES_BOOL}
            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score, score_type="boolean")


def run_case_meta_judge(
    input_text,
    output_text,
    model_name,
    retrieval_results=None,
    output_results=None,
    base_url=None,
):
    case_summary = build_case_judgment_summary(
        retrieval_results=retrieval_results,
        output_results=output_results,
    )

    meta_judge = JudgeAuditMetaJudge(
        model=model_name,
        base_url=base_url,
    )

    meta_result = _normalize_judgment(
        meta_judge.judge(
            input=input_text,
            output=output_text,
            expected=case_summary,
        )
    )

    return {
        "meta_judge": meta_result,
        "case_judgment_summary": case_summary,
        "model": model_name,
    }

def run_output_judges(
    input_text,
    output_text,
    model_name,
    expected_text=None,
    base_url=None,
):
    clean_expected_text = (expected_text or "").strip()

    judges = {
        "context_evaluator": ContextEvaluatorJudge(
            model=model_name,
            base_url=base_url,
        ),
    }

    results = {
        name: _normalize_judgment(
            judge.judge(
                input=input_text,
                output=output_text,
                expected=clean_expected_text,
            )
        )
        for name, judge in judges.items()
    }

    if clean_expected_text:
        classification_judge = ClassificationJudge(
            model=model_name,
            base_url=base_url,
        )

        results["classification_match"] = _normalize_judgment(
            classification_judge.judge(
                input=input_text,
                output=output_text,
                expected=clean_expected_text,
            )
        )

    results["model"] = model_name
    results["control_label"] = clean_expected_text

    return results

def run_kg_comparison_judges(
    patient_note,
    kg_context,
    kg_outputs_by_key,
    no_kg_output,
    model_name,
    control_label=None,
    judge_results_by_output=None,
    base_url=None,
):
    clean_control_label = (control_label or "").strip()
    judge_results_by_output = judge_results_by_output or {}

    comparison_context = dedent(
        f"""
        Patient note:
        {patient_note.strip()}

        Retrieved KG/RAG context:
        {kg_context.strip() if kg_context else "No retrieved KG/RAG context."}
        """
    ).strip()

    improvement_judge = KGImprovementOverNoKGJudge(
        model=model_name,
        base_url=base_url,
    )

    visible_influence_judge = KGVisibleInfluenceJudge(
        model=model_name,
        base_url=base_url,
    )

    results = {}

    for output_key, kg_output in (kg_outputs_by_key or {}).items():
        if not str(kg_output or "").strip():
            continue

        improvement_result = _normalize_judgment(
            improvement_judge.judge(
                input=comparison_context,
                kg_output=kg_output,
                no_kg_output=no_kg_output,
                expected=clean_control_label,
                kg_output_judges=judge_results_by_output.get(output_key),
                no_kg_output_judges=judge_results_by_output.get("no_kg"),
            )
        )

        visible_influence_result = _normalize_judgment(
            visible_influence_judge.judge(
                input=comparison_context,
                kg_output=kg_output,
                no_kg_output=no_kg_output,
                expected=clean_control_label,
            )
        )

        results[output_key] = {
            "label": output_key,
            "kg_improved_over_no_kg": improvement_result,
            "kg_visible_influence": visible_influence_result,
        }

    return {
        "comparisons": results,
        "model": model_name,
        "control_label": clean_control_label,
    }