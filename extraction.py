import json
import os
import re
import requests
import threading
import time
import math
import copy
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from queue import Queue, Empty

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "scrape" / "sundhed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kg_extractions"
DEBUG_DIR = OUTPUT_DIR / "_debug"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

BASE_URLS = [os.getenv("BASE_URL1"), os.getenv("BASE_URL2")]
BASE_URLS = [url for url in BASE_URLS if url]
ENDPOINT = os.getenv("ENDPOINT")

if not BASE_URLS or not ENDPOINT:
    raise ValueError("Missing BASE_URL1/BASE_URL2 and/or ENDPOINT in environment variables")

MODEL_NAME = "gemma3:27b"
NUM_CTX = 30000
TIMEOUT = 600
MAX_PROMPT_TOKENS = 23000

print_lock = threading.Lock()

SYSTEM_PROMPT = """
You are an expert Clinical Knowledge Graph Extractor.

Your task is to extract a clinically meaningful consistent knowledge graph from the text.

All node names, labels, relationship types, and descriptions must be in Danish.
Return only valid JSON matching the required schema.

## EXTRACTION PURPOSE
The knowledge graph is designed to supplement downstream LLM calls used by healthcare professionals.

Do NOT aim to reproduce administrative or organizational content.

## COVERAGE RULES
The text will be dense with information.
- For each sentence ensure that all the clinically meaningful relationships and nodes have been created.
- A sentence might have more meaningful relationships than just one.
- Compare created nodes and edges to the text and ask yourself if this covers all nuances of the text.

## WHAT TO DEPRIORITIZE OR OMIT
Only include these if they add clear clinical meaning:

- administrative instructions
- referral processes
- organizational or system-specific details
- database or registry descriptions
- repeated explanatory filler text

## EXTRACTION STRATEGY
Work systematically before finalizing output:

1. Identify candidate concepts
2. Normalize and merge duplicates
3. Extract meaningful relationships
4. Ensure coverage of all clinical dimensions
5. Validate structure and completeness
6. Compare sentence to relationships and determine if coverage is sufficient

Only output the final validated graph.

## ENTITY EXTRACTION RULES
Extract distinct medical concepts as reusable nodes.

Use clinically meaningful categories such as:
- Sygdom
- Tilstand
- Diagnose
- Symptom
- KliniskFund
- Test
- Procedure
- Behandling
- Medicin
- Medicingruppe
- Mekanisme
- Reaktionstype
- Differentialdiagnose
- Komplikation
- Risikofaktor
- Population
- Tidsforl├©b
- Kontraindikation

You may introduce new types if clinically precise.

Use PascalCase labels.

You may introduce additional clinically meaningful labels if clearly justified by the source.

## NORMALIZATION RULES
1. Use singular form
2. Use standard Danish medical terminology
3. Use natural language (NOT camelCase or PascalCase for node ids)
4. Reuse identical wording consistently
5. Merge duplicates
6. Preserve clinically meaningful distinctions
7. Keep both parent and child concepts when useful
8. Preserve named entities (tests, genes, scores, syndromes, subtypes)

## RELATIONSHIP EXTRACTION RULES
Use SCREAMING_SNAKE_CASE.

Follow clinical direction:

- Sygdom/Tilstand ÔåÆ symptom
- Sygdom/Tilstand ÔåÆ ├Ñrsag
- Sygdom/Tilstand ÔåÆ differentialdiagnose
- Reaktionstype ÔåÆ symptom
- Reaktionstype ÔåÆ mekanisme

Avoid reversed or abstract relationships.

Common types include:

- HAR_SYMPTOM
- HAR_KLINISK_FUND
- UDL├ÿSES_AF
- FOR├àRSAGES_AF
- HAR_MEKANISME
- MEDIERES_AF
- HAR_REAKTIONSTYPE
- ER_EN_TYPE_AF
- ER_ASSOCIERET_MED
- HAR_RISIKOFAKTOR
- DIAGNOSTICERES_MED
- KAN_BEKR├åFTES_MED
- KAN_IKKE_UDELUKKES_MED
- HAR_BEGR├åNSNING
- SKAL_ADSKILLES_FRA
- HAR_DIFFERENTIALDIAGNOSE
- BEHANDLES_MED
- KAN_MEDF├ÿRE
- KAN_MANIFESTERE_SIG_SOM
- ER_KONTRAINDICERET_VED
- FOREKOMMER_HOS
- DEBUTERER_INDEN_FOR
- VARER_TYPISK
- KAN_HAVE_KRYDSREAKTION_MED

You may introduce new types if clinically precise.

## STRUCTURAL RULES
- No duplicate nodes
- No duplicate relationships
- Avoid isolated important nodes
- Connect related concepts (hierarchy, mechanism, subtype)

## LIST EXTRACTION RULE
When the text includes lists of items extract key items individually.

Lists MUST be fully represented in the final output.

Do NOT collapse important lists into one vague node.

## HIERARCHY RULE
When a disease belongs to a broader category, include both and connect:

Subtype ÔåÆ ER_EN_TYPE_AF ÔåÆ Parent

## OUTPUT STRUCTURE

{
  "nodes": [
    {
      "id": "string",
      "label": "string",
      "description": "string"
    }
  ],
  "relationships": [
    {
      "source": "string",
      "target": "string",
      "type": "string",
      "description": "string"
    }
  ]
}

## VALIDATION RULES
- Every relationship must reference existing nodes
- Output must be valid JSON
- Do not output any text outside JSON

## EXAMPLES

### Example 1: basic disease, symptoms, and treatment
Input:
"Influenza giver feber og hoste. Tilstanden kan behandles med antivirale l├ªgemidler."

Output:
{
  "nodes": [
    {"id": "Influenza", "label": "Sygdom", "description": "Akut virusinfektion i luftvejene."},
    {"id": "Feber", "label": "Symptom", "description": "Forh├©jet kropstemperatur."},
    {"id": "Hoste", "label": "Symptom", "description": "Refleks fra luftvejene."},
    {"id": "Antiviralt l├ªgemiddel", "label": "Medicin", "description": "Medicin mod virusinfektioner."}
  ],
  "relationships": [
    {"source": "Influenza", "target": "Feber", "type": "HAR_SYMPTOM", "description": "Influenza kan give feber."},
    {"source": "Influenza", "target": "Hoste", "type": "HAR_SYMPTOM", "description": "Influenza kan give hoste."},
    {"source": "Influenza", "target": "Antiviralt l├ªgemiddel", "type": "BEHANDLES_MED", "description": "Influenza behandles med antivirale l├ªgemidler."}
  ]
}

### Example 2: negative test does not exclude disease
Input:
"Negativ specifikt IgE udelukker ikke penicillinallergi."

Output:
{
  "nodes": [
    {"id": "Penicillinallergi", "label": "Tilstand", "description": "Allergisk reaktion mod penicillin."},
    {"id": "Specifikt IgE for penicillin", "label": "Test", "description": "Blodpr├©ve for IgE mod penicillin."},
    {"id": "Negativ specifikt IgE for penicillin", "label": "KliniskFund", "description": "Negativt resultat af specifikt IgE for penicillin."}
  ],
  "relationships": [
    {
      "source": "Negativ specifikt IgE for penicillin",
      "target": "Penicillinallergi",
      "type": "KAN_IKKE_UDELUKKES_MED",
      "description": "Negativ specifikt IgE udelukker ikke penicillinallergi."
    }
  ]
}

### Example 3: symptom in a population context
Input:
"NSAID kan udl├©se bronkospasme hos patienter med astma."

Output:
{
  "nodes": [
    {"id": "NSAID", "label": "Medicingruppe", "description": "Ikke-steroide antiinflammatoriske l├ªgemidler."},
    {"id": "Bronkospasme", "label": "KliniskFund", "description": "Sammentr├ªkning af bronkier."},
    {"id": "Astma", "label": "Sygdom", "description": "Kronisk inflammatorisk luftvejssygdom."}
  ],
  "relationships": [
    {"source": "NSAID", "target": "Bronkospasme", "type": "KAN_MEDF├ÿRE", "description": "NSAID kan udl├©se bronkospasme."},
    {"source": "Bronkospasme", "target": "Astma", "type": "FOREKOMMER_HOS", "description": "Bronkospasme forekommer hos patienter med astma."}
  ]
}

### Example 4: hierarchy, mechanism, diagnosis, treatment, prognosis
Input:
"Trombotisk trombocytopenisk purpura er en trombotisk mikroangiopati. Diagnosen mist├ªnkes ved trombocytopeni og h├ªmolyse og bekr├ªftes ved ADAMTS13-aktivitet under 10 %. Autoimmun TTP skyldes autoantistoffer mod ADAMTS13. HUS er en vigtig differentialdiagnose. Behandlingen er plasmaferese og glukokortikoid. Ubehandlet er mortaliteten h├©j."

Output:
{
  "nodes": [
    {"id": "Trombotisk trombocytopenisk purpura", "label": "Sygdom", "description": "Sj├ªlden trombotisk mikroangiopati."},
    {"id": "Trombotisk mikroangiopati", "label": "Sygdom", "description": "Sygdomsgruppe med mikrotromboser, trombocytopeni og h├ªmolyse."},
    {"id": "Trombocytopeni", "label": "KliniskFund", "description": "Lavt antal trombocytter."},
    {"id": "H├ªmolyse", "label": "KliniskFund", "description": "Destruktion af r├©de blodlegemer."},
    {"id": "ADAMTS13-aktivitet under 10 %", "label": "KliniskFund", "description": "Sv├ªrt nedsat ADAMTS13-aktivitet."},
    {"id": "Autoimmun TTP", "label": "Sygdom", "description": "Erhvervet autoimmun form af trombotisk trombocytopenisk purpura."},
    {"id": "Autoantistof mod ADAMTS13", "label": "Mekanisme", "description": "Autoimmun antistofdannelse mod ADAMTS13."},
    {"id": "H├ªmolytisk ur├ªmisk syndrom", "label": "Differentialdiagnose", "description": "Vigtig differentialdiagnose til trombotisk trombocytopenisk purpura."},
    {"id": "Plasmaferese", "label": "Behandling", "description": "Plasmaudskiftningsbehandling."},
    {"id": "Glukokortikoid", "label": "Medicin", "description": "Immund├ªmpende behandling."},
    {"id": "H├©j mortalitet uden behandling", "label": "Komplikation", "description": "Ubehandlet sygdom har h├©j d├©delighed."}
  ],
  "relationships": [
    {"source": "Trombotisk trombocytopenisk purpura", "target": "Trombotisk mikroangiopati", "type": "ER_EN_TYPE_AF", "description": "Trombotisk trombocytopenisk purpura er en type trombotisk mikroangiopati."},
    {"source": "Trombotisk trombocytopenisk purpura", "target": "Trombocytopeni", "type": "HAR_KLINISK_FUND", "description": "Trombotisk trombocytopenisk purpura er forbundet med trombocytopeni."},
    {"source": "Trombotisk trombocytopenisk purpura", "target": "H├ªmolyse", "type": "HAR_KLINISK_FUND", "description": "Trombotisk trombocytopenisk purpura er forbundet med h├ªmolyse."},
    {"source": "Trombotisk trombocytopenisk purpura", "target": "ADAMTS13-aktivitet under 10 %", "type": "KAN_BEKR├åFTES_MED", "description": "Trombotisk trombocytopenisk purpura kan bekr├ªftes ved ADAMTS13-aktivitet under 10 %."},
    {"source": "Autoimmun TTP", "target": "Trombotisk trombocytopenisk purpura", "type": "ER_EN_TYPE_AF", "description": "Autoimmun TTP er en type trombotisk trombocytopenisk purpura."},
    {"source": "Autoimmun TTP", "target": "Autoantistof mod ADAMTS13", "type": "FOR├àRSAGES_AF", "description": "Autoimmun TTP skyldes autoantistoffer mod ADAMTS13."},
    {"source": "Trombotisk trombocytopenisk purpura", "target": "H├ªmolytisk ur├ªmisk syndrom", "type": "HAR_DIFFERENTIALDIAGNOSE", "description": "H├ªmolytisk ur├ªmisk syndrom er en vigtig differentialdiagnose til trombotisk trombocytopenisk purpura."},
    {"source": "Trombotisk trombocytopenisk purpura", "target": "Plasmaferese", "type": "BEHANDLES_MED", "description": "Trombotisk trombocytopenisk purpura behandles med plasmaferese."},
    {"source": "Trombotisk trombocytopenisk purpura", "target": "Glukokortikoid", "type": "BEHANDLES_MED", "description": "Trombotisk trombocytopenisk purpura behandles med glukokortikoid."},
    {"source": "Trombotisk trombocytopenisk purpura", "target": "H├©j mortalitet uden behandling", "type": "KAN_MEDF├ÿRE", "description": "Ubehandlet trombotisk trombocytopenisk purpura har h├©j mortalitet."}
  ]
}

### Example 5: diagnostic threshold, repeat logic, and measurement limitation
Input:
"Diagnosen hypothyreose stilles ved gentagen h├©j TSH og lav frit T4. Subklinisk hypothyreose defineres som h├©j TSH og normal frit T4. Ved akut sygdom skal TSH og T4 tolkes varsomt."

Output:
{
  "nodes": [
    {"id": "Hypothyreose", "label": "Sygdom", "description": "Tilstand med nedsat thyreoideafunktion."},
    {"id": "Subklinisk hypothyreose", "label": "Tilstand", "description": "Mild hypothyreose med forh├©jet TSH og normal frit T4."},
    {"id": "TSH", "label": "Test", "description": "Blodpr├©ve til vurdering af thyreoideafunktion."},
    {"id": "Frit T4", "label": "Test", "description": "Blodpr├©ve til vurdering af thyroxin."},
    {"id": "Gentagen h├©j TSH", "label": "KliniskFund", "description": "Vedvarende forh├©jet TSH ved gentagne m├Ñlinger."},
    {"id": "Lav frit T4", "label": "KliniskFund", "description": "Nedsat frit T4."},
    {"id": "Normal frit T4", "label": "KliniskFund", "description": "Normalt frit T4."},
    {"id": "Akut sygdom", "label": "Tilstand", "description": "Akut medicinsk tilstand som kan p├Ñvirke tolkning af blodpr├©ver."}
  ],
  "relationships": [
    {"source": "Hypothyreose", "target": "Gentagen h├©j TSH", "type": "KAN_BEKR├åFTES_MED", "description": "Hypothyreose bekr├ªftes ved gentagen h├©j TSH."},
    {"source": "Hypothyreose", "target": "Lav frit T4", "type": "KAN_BEKR├åFTES_MED", "description": "Hypothyreose bekr├ªftes ved lav frit T4."},
    {"source": "Subklinisk hypothyreose", "target": "Hypothyreose", "type": "ER_EN_TYPE_AF", "description": "Subklinisk hypothyreose er en type hypothyreose."},
    {"source": "Subklinisk hypothyreose", "target": "Gentagen h├©j TSH", "type": "KAN_BEKR├åFTES_MED", "description": "Subklinisk hypothyreose er forbundet med gentagen h├©j TSH."},
    {"source": "Subklinisk hypothyreose", "target": "Normal frit T4", "type": "KAN_BEKR├åFTES_MED", "description": "Subklinisk hypothyreose er forbundet med normal frit T4."},
    {"source": "Akut sygdom", "target": "TSH", "type": "HAR_BEGR├åNSNING", "description": "Akut sygdom kan begr├ªnse tolkningen af TSH."},
    {"source": "Akut sygdom", "target": "Frit T4", "type": "HAR_BEGR├åNSNING", "description": "Akut sygdom kan begr├ªnse tolkningen af frit T4."}
  ]
}

### Example 6: broader hierarchy and explicit subtype structure
Input:
"Akut leuk├ªmi opdeles i akut myeloid leuk├ªmi og akut lymfatisk leuk├ªmi."

Output:
{
  "nodes": [
    {"id": "Akut leuk├ªmi", "label": "Sygdom", "description": "Hurtigt udviklende kr├ªftsygdom i blod og knoglemarv."},
    {"id": "Akut myeloid leuk├ªmi", "label": "Sygdom", "description": "Myeloid form for akut leuk├ªmi."},
    {"id": "Akut lymfatisk leuk├ªmi", "label": "Sygdom", "description": "Lymfatisk form for akut leuk├ªmi."}
  ],
  "relationships": [
    {"source": "Akut myeloid leuk├ªmi", "target": "Akut leuk├ªmi", "type": "ER_EN_TYPE_AF", "description": "Akut myeloid leuk├ªmi er en type akut leuk├ªmi."},
    {"source": "Akut lymfatisk leuk├ªmi", "target": "Akut leuk├ªmi", "type": "ER_EN_TYPE_AF", "description": "Akut lymfatisk leuk├ªmi er en type akut leuk├ªmi."}
  ]
}

### Example 7: risk factor versus mechanism
Input:
"Rygning ├©ger risikoen for lungekr├ªft, som er drevet af genetiske mutationer."

Output:
{
  "nodes": [
    {"id": "Lungekr├ªft", "label": "Sygdom", "description": "Malign sygdom i lungerne."},
    {"id": "Rygning", "label": "Risikofaktor", "description": "Eksponering for tobaksr├©g."},
    {"id": "Genetisk mutation", "label": "Mekanisme", "description": "Genetisk forandring der kan drive kr├ªftudvikling."}
  ],
  "relationships": [
    {"source": "Lungekr├ªft", "target": "Rygning", "type": "HAR_RISIKOFAKTOR", "description": "Rygning ├©ger risikoen for lungekr├ªft."},
    {"source": "Lungekr├ªft", "target": "Genetisk mutation", "type": "HAR_MEKANISME", "description": "Lungekr├ªft er drevet af genetiske mutationer."}
  ]
}

### Example 8: contraindication
Input:
"Metformin er kontraindiceret ved sv├ªr nyreinsufficiens."

Output:
{
  "nodes": [
    {"id": "Metformin", "label": "Medicin", "description": "Blodsukkers├ªnkende l├ªgemiddel."},
    {"id": "Sv├ªr nyreinsufficiens", "label": "Tilstand", "description": "Alvorligt nedsat nyrefunktion."}
  ],
  "relationships": [
    {"source": "Metformin", "target": "Sv├ªr nyreinsufficiens", "type": "ER_KONTRAINDICERET_VED", "description": "Metformin er kontraindiceret ved sv├ªr nyreinsufficiens."}
  ]
}

### Example 9: monitoring and prognosis
Input:
"Patienter med hjertesvigt monitoreres med BNP. Forh├©jet BNP er associeret med d├Ñrlig prognose."

Output:
{
  "nodes": [
    {"id": "Hjertesvigt", "label": "Sygdom", "description": "Tilstand med nedsat pumpefunktion af hjertet."},
    {"id": "BNP", "label": "Test", "description": "Blodpr├©ve der afspejler hjertets belastning."},
    {"id": "Forh├©jet BNP", "label": "KliniskFund", "description": "Forh├©jet BNP-niveau."},
    {"id": "D├Ñrlig prognose", "label": "Tilstand", "description": "Forbundet med ├©get risiko for forv├ªrring eller d├©d."}
  ],
  "relationships": [
    {"source": "Hjertesvigt", "target": "BNP", "type": "DIAGNOSTICERES_MED", "description": "BNP anvendes ved monitorering af hjertesvigt."},
    {"source": "Forh├©jet BNP", "target": "D├Ñrlig prognose", "type": "ER_ASSOCIERET_MED", "description": "Forh├©jet BNP er associeret med d├Ñrlig prognose."}
  ]
}

### Example 10: complication list should be unpacked
Input:
"Diabetes kan f├©re til nefropati og neuropati."

Output:
{
  "nodes": [
    {"id": "Diabetes", "label": "Sygdom", "description": "Kronisk metabolisk sygdom med forh├©jet blodsukker."},
    {"id": "Nefropati", "label": "Komplikation", "description": "Nyreskade som f├©lge af sygdom."},
    {"id": "Neuropati", "label": "Komplikation", "description": "Nerveskade som f├©lge af sygdom."}
  ],
  "relationships": [
    {"source": "Diabetes", "target": "Nefropati", "type": "KAN_MEDF├ÿRE", "description": "Diabetes kan f├©re til nefropati."},
    {"source": "Diabetes", "target": "Neuropati", "type": "KAN_MEDF├ÿRE", "description": "Diabetes kan f├©re til neuropati."}
  ]
}

### Example 11: named subtype, named mechanism, special treatment, special prognosis
Input:
"Akut promyelocytleuk├ªmi er en undertype af akut myeloid leuk├ªmi. Sygdommen er karakteriseret ved PML-RARA-fusion og behandles med all-trans-retinoinsyre og arsentrioxid. Prognosen er meget god ved opn├Ñet sygdomskontrol."

Output:
{
  "nodes": [
    {"id": "Akut promyelocytleuk├ªmi", "label": "Sygdom", "description": "Undertype af akut myeloid leuk├ªmi."},
    {"id": "Akut myeloid leuk├ªmi", "label": "Sygdom", "description": "Akut myeloid leuk├ªmi."},
    {"id": "PML-RARA-fusion", "label": "Mekanisme", "description": "Karakteristisk genetisk forandring ved akut promyelocytleuk├ªmi."},
    {"id": "All-trans-retinoinsyre", "label": "Medicin", "description": "Differentierende behandling ved akut promyelocytleuk├ªmi."},
    {"id": "Arsentrioxid", "label": "Medicin", "description": "Behandling ved akut promyelocytleuk├ªmi."},
    {"id": "God prognose ved sygdomskontrol", "label": "Tilstand", "description": "Meget god prognose ved opn├Ñet sygdomskontrol."}
  ],
  "relationships": [
    {"source": "Akut promyelocytleuk├ªmi", "target": "Akut myeloid leuk├ªmi", "type": "ER_EN_TYPE_AF", "description": "Akut promyelocytleuk├ªmi er en type akut myeloid leuk├ªmi."},
    {"source": "Akut promyelocytleuk├ªmi", "target": "PML-RARA-fusion", "type": "HAR_MEKANISME", "description": "Akut promyelocytleuk├ªmi er karakteriseret ved PML-RARA-fusion."},
    {"source": "Akut promyelocytleuk├ªmi", "target": "All-trans-retinoinsyre", "type": "BEHANDLES_MED", "description": "Akut promyelocytleuk├ªmi behandles med all-trans-retinoinsyre."},
    {"source": "Akut promyelocytleuk├ªmi", "target": "Arsentrioxid", "type": "BEHANDLES_MED", "description": "Akut promyelocytleuk├ªmi behandles med arsentrioxid."},
    {"source": "Akut promyelocytleuk├ªmi", "target": "God prognose ved sygdomskontrol", "type": "ER_ASSOCIERET_MED", "description": "Akut promyelocytleuk├ªmi har god prognose ved opn├Ñet sygdomskontrol."}
  ]
}

### Example 12: explicit diagnostic list should be unpacked
Input:
"Ved mistanke om akut myeloid leuk├ªmi anvendes knoglemarvsbiopsi, flowcytometrisk mark├©ranalyse og cytogenetiske unders├©gelser."

Output:
{
  "nodes": [
    {"id": "Akut myeloid leuk├ªmi", "label": "Sygdom", "description": "Akut leuk├ªmi med myeloid differentiering."},
    {"id": "Knoglemarvsbiopsi", "label": "Procedure", "description": "V├ªvspr├©ve fra knoglemarven."},
    {"id": "Flowcytometrisk mark├©ranalyse", "label": "Test", "description": "Analyse af cellemark├©rer ved flowcytometri."},
    {"id": "Cytogenetisk unders├©gelse", "label": "Test", "description": "Unders├©gelse af kromosomforandringer."}
  ],
  "relationships": [
    {"source": "Akut myeloid leuk├ªmi", "target": "Knoglemarvsbiopsi", "type": "DIAGNOSTICERES_MED", "description": "Akut myeloid leuk├ªmi udredes med knoglemarvsbiopsi."},
    {"source": "Akut myeloid leuk├ªmi", "target": "Flowcytometrisk mark├©ranalyse", "type": "DIAGNOSTICERES_MED", "description": "Akut myeloid leuk├ªmi udredes med flowcytometrisk mark├©ranalyse."},
    {"source": "Akut myeloid leuk├ªmi", "target": "Cytogenetisk unders├©gelse", "type": "DIAGNOSTICERES_MED", "description": "Akut myeloid leuk├ªmi udredes med cytogenetisk unders├©gelse."}
  ]
}
""".strip()

def log_line(message: str):
    with print_lock:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {message}", flush=True)


def _endpoint_label(base_url: str) -> str:
    for i, url in enumerate(BASE_URLS, start=1):
        if url == base_url:
            return f"endpoint_{i}"
    return "endpoint_unknown"


def _compact_text_preview(user_payload: dict, max_len: int = 160) -> str:
    title = (user_payload.get("title") or "").strip()
    sections = user_payload.get("sections") or []

    first_text = ""
    for section in sections:
        content = section.get("content") or []
        for item in content:
            s = str(item).strip()
            if s:
                first_text = s
                break
        if first_text:
            break

    preview = first_text or title or "[ingen tekst fundet]"
    preview = re.sub(r"\s+", " ", preview).strip()

    if len(preview) > max_len:
        preview = preview[:max_len - 3] + "..."

    return preview


def _estimate_tokens_from_text(text: str) -> int:
    text = text or ""
    return max(1, round(len(text) / 4))


def _estimate_prompt_tokens(system_prompt: str, user_payload: dict) -> int:
    return (
        _estimate_tokens_from_text(system_prompt)
        + _estimate_tokens_from_text(json.dumps(user_payload, ensure_ascii=False))
    )


def llm_chat(system_prompt: str, user_json_text: str, base_url: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_json_text},
        ],
        "options": {"num_ctx": NUM_CTX},
    }

    r = requests.post(f"{base_url}{ENDPOINT}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()

    try:
        data = r.json()
    except Exception as e:
        raise ValueError(f"LLM response was not valid JSON: {repr(e)} | Raw response: {r.text[:3000]}")

    content = None

    if isinstance(data, dict):
        if "choices" in data:
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                if isinstance(message, dict):
                    content = message.get("content")
                if not content:
                    content = choices[0].get("text")

        if not content and "message" in data:
            message = data.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content")

        if not content and "response" in data:
            content = data.get("response")

    if content is None:
        raise ValueError(
            f"LLM returned no parsable content. Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    content = str(content).strip()

    if not content:
        raise ValueError(
            f"LLM returned empty content. Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    usage = data.get("usage") if isinstance(data, dict) else None

    if not usage and isinstance(data, dict):
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")

        if prompt_tokens is not None or completion_tokens is not None:
            usage = {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
            }

    if not usage:
        estimated_prompt_tokens = (
            _estimate_tokens_from_text(system_prompt)
            + _estimate_tokens_from_text(user_json_text)
        )
        estimated_completion_tokens = _estimate_tokens_from_text(content)

        usage = {
            "prompt_tokens": estimated_prompt_tokens,
            "completion_tokens": estimated_completion_tokens,
            "total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
            "estimated": True,
        }

    return {
        "content": content,
        "usage": usage or {},
        "raw_response": data,
    }


def _save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _safe_title_slug(title: str, fallback: str = "untitled", max_length: int = 120) -> str:
    text = (title or fallback).strip().lower()
    text = text.replace("├ª", "ae").replace("├©", "oe").replace("├Ñ", "aa")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_length] or fallback


def _extract_json(text: str) -> dict:
    t = (text or "").strip()

    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    start = t.find("{")
    end = t.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object")

    candidate = t[start:end + 1]
    return json.loads(candidate)


def _add_missing_nodes(data: dict) -> tuple[dict, list[str]]:
    if not isinstance(data, dict):
        return data, []

    nodes = data.get("nodes", [])
    relationships = data.get("relationships", [])

    existing_ids = {
        node["id"]
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str) and node.get("id").strip()
    }

    referenced_ids = set()

    for rel in relationships:
        if not isinstance(rel, dict):
            continue

        source = rel.get("source")
        target = rel.get("target")

        if isinstance(source, str) and source.strip():
            referenced_ids.add(source)
        if isinstance(target, str) and target.strip():
            referenced_ids.add(target)

    missing_ids = sorted(referenced_ids - existing_ids)

    for missing_id in missing_ids:
        nodes.append({
            "id": missing_id,
            "label": "Missing",
            "description": "Denne node blev automatisk oprettet, fordi den blev brugt i en relation uden at v├ªre defineret."
        })

    data["nodes"] = nodes
    return data, missing_ids


def _validate_kg_json(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Output is not a JSON object")

    if set(data.keys()) != {"nodes", "relationships"}:
        raise ValueError("Top-level keys must be exactly 'nodes' and 'relationships'")

    if not isinstance(data["nodes"], list):
        raise ValueError("'nodes' must be a list")

    if not isinstance(data["relationships"], list):
        raise ValueError("'relationships' must be a list")

    node_ids = set()

    for i, node in enumerate(data["nodes"]):
        if not isinstance(node, dict):
            raise ValueError(f"Node at index {i} is not an object")

        if set(node.keys()) != {"id", "label", "description"}:
            raise ValueError(f"Node at index {i} must contain exactly id, label, description")

        if not all(isinstance(node[k], str) and node[k].strip() for k in ["id", "label", "description"]):
            raise ValueError(f"Node at index {i} has empty required fields")

        if node["id"] in node_ids:
            raise ValueError(f"Duplicate node id at index {i}: {node['id']}")

        node_ids.add(node["id"])

    relationship_keys = set()

    for i, rel in enumerate(data["relationships"]):
        if not isinstance(rel, dict):
            raise ValueError(f"Relationship at index {i} is not an object")

        if set(rel.keys()) != {"source", "target", "type", "description"}:
            raise ValueError(f"Relationship at index {i} must contain exactly source, target, type, description")

        if not all(isinstance(rel[k], str) and rel[k].strip() for k in ["source", "target", "type", "description"]):
            raise ValueError(f"Relationship at index {i} has empty required fields")

        if rel["source"] not in node_ids:
            raise ValueError(f"Unknown relationship source at index {i}: {rel['source']}")

        if rel["target"] not in node_ids:
            raise ValueError(f"Unknown relationship target at index {i}: {rel['target']}")

        rel_key = (rel["source"], rel["target"], rel["type"], rel["description"])
        if rel_key in relationship_keys:
            raise ValueError(f"Duplicate relationship at index {i}: {rel_key}")

        relationship_keys.add(rel_key)

    return data


def _flatten_section_content(content):
    flat = []

    for item in content or []:
        if isinstance(item, list):
            for sub in item:
                s = str(sub).strip()
                if s:
                    flat.append(s)
        else:
            s = str(item).strip()
            if s:
                flat.append(s)

    return flat


def _is_noise_section(section: dict) -> bool:
    heading = (section.get("heading") or "").strip().lower()

    noise_terms = [
        "kilder",
        "referencer",
        "patientorganisation",
        "patientorganisationer",
        "b├©ger",
        "boeger",
        "link",
        "kommentar",
    ]

    return any(term in heading for term in noise_terms)


def _build_user_payload(page: dict) -> dict:
    title = page.get("title", "").strip()
    sections = page.get("sections", [])

    if sections:
        sections = sections[1:]

    cleaned_sections = []
    for section in sections:
        if not section or _is_noise_section(section):
            continue

        content = _flatten_section_content(section.get("content", []))
        if not content:
            continue

        cleaned_sections.append({
            "heading": section.get("heading"),
            "content": content
        })

    return {
        "title": title,
        "sections": cleaned_sections
    }


def _split_text_roughly(text: str, parts: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    sentence_chunks = re.split(r'(?<=[.!?])\s+', text)

    if len(sentence_chunks) >= parts:
        chunk_size = math.ceil(len(sentence_chunks) / parts)
        return [
            " ".join(sentence_chunks[i:i + chunk_size]).strip()
            for i in range(0, len(sentence_chunks), chunk_size)
            if " ".join(sentence_chunks[i:i + chunk_size]).strip()
        ]

    word_chunks = text.split()
    if len(word_chunks) >= parts:
        chunk_size = math.ceil(len(word_chunks) / parts)
        return [
            " ".join(word_chunks[i:i + chunk_size]).strip()
            for i in range(0, len(word_chunks), chunk_size)
            if " ".join(word_chunks[i:i + chunk_size]).strip()
        ]

    return [text]


def _chunk_user_payload(user_payload: dict, system_prompt: str, max_prompt_tokens: int) -> list[dict]:
    estimated_prompt_tokens = _estimate_prompt_tokens(system_prompt, user_payload)

    if estimated_prompt_tokens <= max_prompt_tokens:
        return [user_payload]

    min_chunks = math.ceil(estimated_prompt_tokens / max_prompt_tokens)

    title = user_payload.get("title", "")
    sections = user_payload.get("sections", []) or []

    def make_payload(section_subset: list[dict]) -> dict:
        return {
            "title": title,
            "sections": section_subset
        }

    def payload_tokens(section_subset: list[dict]) -> int:
        return _estimate_prompt_tokens(system_prompt, make_payload(section_subset))

    chunks = []
    current_sections = []

    def flush_current():
        nonlocal current_sections
        if current_sections:
            chunks.append(make_payload(current_sections))
            current_sections = []

    for section in sections:
        section_copy = copy.deepcopy(section)

        if payload_tokens(current_sections + [section_copy]) <= max_prompt_tokens:
            current_sections.append(section_copy)
            continue

        section_content = section_copy.get("content", [])

        if not section_content:
            if current_sections:
                flush_current()
            current_sections.append(section_copy)
            continue

        if payload_tokens([section_copy]) <= max_prompt_tokens:
            if current_sections:
                flush_current()
            current_sections.append(section_copy)
            continue

        if current_sections:
            flush_current()

        split_section_parts = []
        current_part_content = []

        for item in section_content:
            tentative_section = {
                "heading": section_copy.get("heading"),
                "content": current_part_content + [item]
            }

            if payload_tokens([tentative_section]) <= max_prompt_tokens:
                current_part_content.append(item)
                continue

            if current_part_content:
                split_section_parts.append({
                    "heading": section_copy.get("heading"),
                    "content": current_part_content
                })
                current_part_content = []

            single_item_section = {
                "heading": section_copy.get("heading"),
                "content": [item]
            }

            if payload_tokens([single_item_section]) <= max_prompt_tokens:
                current_part_content = [item]
                continue

            if isinstance(item, str):
                item_est = payload_tokens([{
                    "heading": section_copy.get("heading"),
                    "content": [item]
                }])
                subparts_needed = max(2, math.ceil(item_est / max_prompt_tokens))
                split_texts = _split_text_roughly(item, subparts_needed)

                for split_text in split_texts:
                    split_item_section = {
                        "heading": section_copy.get("heading"),
                        "content": [split_text]
                    }

                    if payload_tokens([split_item_section]) > max_prompt_tokens:
                        words = split_text.split()
                        chunk_size = max(1, len(words) // 2)
                        smaller_parts = [
                            " ".join(words[i:i + chunk_size]).strip()
                            for i in range(0, len(words), chunk_size)
                            if " ".join(words[i:i + chunk_size]).strip()
                        ]
                        for smaller in smaller_parts:
                            split_section_parts.append({
                                "heading": section_copy.get("heading"),
                                "content": [smaller]
                            })
                    else:
                        split_section_parts.append({
                            "heading": section_copy.get("heading"),
                            "content": [split_text]
                        })
            else:
                split_section_parts.append({
                    "heading": section_copy.get("heading"),
                    "content": [item]
                })

        if current_part_content:
            split_section_parts.append({
                "heading": section_copy.get("heading"),
                "content": current_part_content
            })

        for part in split_section_parts:
            if current_sections and payload_tokens(current_sections + [part]) <= max_prompt_tokens:
                current_sections.append(part)
            else:
                if current_sections:
                    flush_current()
                current_sections.append(part)

    if current_sections:
        flush_current()

    if len(chunks) > min_chunks:
        merged = []
        i = 0
        while i < len(chunks):
            current = copy.deepcopy(chunks[i])
            j = i + 1

            while j < len(chunks):
                candidate_sections = current["sections"] + chunks[j]["sections"]
                if payload_tokens(candidate_sections) <= max_prompt_tokens:
                    current["sections"] = candidate_sections
                    j += 1
                else:
                    break

            merged.append(current)
            i = j

        chunks = merged

    return chunks


def _output_path_for_input(input_path: Path, chunk_index: int | None = None) -> Path:
    base_slug = _safe_title_slug(input_path.stem, fallback="untitled")
    suffix = "" if chunk_index in (None, 0) else f"_{chunk_index}"
    return OUTPUT_DIR / f"kg-{base_slug}{suffix}.json"


def _debug_path_for_input(input_path: Path, chunk_index: int | None = None) -> Path:
    base_slug = _safe_title_slug(input_path.stem, fallback="untitled")
    suffix = "" if chunk_index in (None, 0) else f"_{chunk_index}"
    return DEBUG_DIR / f"raw-{base_slug}{suffix}.txt"


def _failed_output_path_for_input(input_path: Path, chunk_index: int | None = None) -> Path:
    base_slug = _safe_title_slug(input_path.stem, fallback="untitled")
    suffix = "" if chunk_index in (None, 0) else f"_{chunk_index}"
    return OUTPUT_DIR / f"kg-{base_slug}{suffix}.txt"


def _process_single_payload(
    input_path: Path,
    user_payload: dict,
    base_url: str,
    chunk_index: int = 0,
    total_chunks: int = 1
) -> dict:
    started = time.perf_counter()

    output_path = _output_path_for_input(input_path, chunk_index)
    debug_path = _debug_path_for_input(input_path, chunk_index)
    failed_output_path = _failed_output_path_for_input(input_path, chunk_index)

    endpoint_name = _endpoint_label(base_url)
    preview = _compact_text_preview(user_payload)
    estimated_prompt_tokens = _estimate_prompt_tokens(SYSTEM_PROMPT, user_payload)

    chunk_info = f" | chunk={chunk_index + 1}/{total_chunks}" if total_chunks > 1 else ""

    log_line(
        f"[START] {endpoint_name} | {input_path.name}{chunk_info} | "
        f"prompt_est={estimated_prompt_tokens} | preview: {preview}"
    )

    try:
        raw_result = llm_chat(
            SYSTEM_PROMPT,
            json.dumps(user_payload, ensure_ascii=False),
            base_url=base_url
        )

        raw = raw_result["content"]
        usage = raw_result.get("usage", {}) or {}

        _save_text(debug_path, raw)

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        data = _extract_json(raw)
        data, placeholder_nodes = _add_missing_nodes(data)
        data = _validate_kg_json(data)

        _save_text(output_path, json.dumps(data, ensure_ascii=False, indent=2))

        duration = time.perf_counter() - started

        log_line(
            f"[DONE]  {endpoint_name} | {input_path.name}{chunk_info} | "
            f"status=json_success | nodes={len(data.get('nodes', []))} | "
            f"rels={len(data.get('relationships', []))} | "
            f"tokens={total_tokens if total_tokens is not None else '?'} "
            f"(prompt={prompt_tokens if prompt_tokens is not None else '?'}, "
            f"completion={completion_tokens if completion_tokens is not None else '?'}) | "
            f"time={duration:.1f}s"
        )

        return {
            "input": str(input_path),
            "output": str(output_path),
            "debug_raw": str(debug_path),
            "status": "json_success",
            "nodes": len(data.get("nodes", [])),
            "relationships": len(data.get("relationships", [])),
            "placeholder_nodes_added": placeholder_nodes,
            "model": MODEL_NAME,
            "base_url": base_url,
            "endpoint_name": endpoint_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_seconds": round(duration, 2),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "extracted_at": datetime.now().isoformat(),
        }

    except Exception as e:
        duration = time.perf_counter() - started

        raw = locals().get("raw", "")
        prompt_tokens = locals().get("prompt_tokens")
        completion_tokens = locals().get("completion_tokens")
        total_tokens = locals().get("total_tokens")

        if raw:
            _save_text(failed_output_path, raw)

        log_line(
            f"[FAIL]  {endpoint_name} | {input_path.name}{chunk_info} | "
            f"status=fallback_txt | tokens={total_tokens if total_tokens is not None else '?'} "
            f"(prompt={prompt_tokens if prompt_tokens is not None else '?'}, "
            f"completion={completion_tokens if completion_tokens is not None else '?'}) | "
            f"time={duration:.1f}s | error={repr(e)}"
        )

        return {
            "input": str(input_path),
            "output": str(failed_output_path),
            "debug_raw": str(debug_path),
            "status": "fallback_txt",
            "nodes": 0,
            "relationships": 0,
            "placeholder_nodes_added": [],
            "model": MODEL_NAME,
            "base_url": base_url,
            "endpoint_name": endpoint_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_seconds": round(duration, 2),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "extracted_at": datetime.now().isoformat(),
        }


def extract_from_file(input_path: Path, base_url: str) -> dict:
    page = json.loads(input_path.read_text(encoding="utf-8"))
    user_payload = _build_user_payload(page)

    estimated_prompt_tokens = _estimate_prompt_tokens(SYSTEM_PROMPT, user_payload)

    if estimated_prompt_tokens <= MAX_PROMPT_TOKENS:
        result = _process_single_payload(
            input_path=input_path,
            user_payload=user_payload,
            base_url=base_url,
            chunk_index=0,
            total_chunks=1
        )
        result["chunked"] = False
        result["original_prompt_est"] = estimated_prompt_tokens
        result["min_chunks"] = 1
        result["actual_chunks"] = 1
        return result

    endpoint_name = _endpoint_label(base_url)
    min_chunks = math.ceil(estimated_prompt_tokens / MAX_PROMPT_TOKENS)
    chunks = _chunk_user_payload(user_payload, SYSTEM_PROMPT, MAX_PROMPT_TOKENS)

    log_line(
        f"[RETRY] {endpoint_name} | {input_path.name} | "
        f"reason=prompt_limit | original_prompt_est={estimated_prompt_tokens} | "
        f"min_chunks={min_chunks} | actual_chunks={len(chunks)}"
    )

    chunk_results = []
    for idx, chunk_payload in enumerate(chunks):
        chunk_result = _process_single_payload(
            input_path=input_path,
            user_payload=chunk_payload,
            base_url=base_url,
            chunk_index=idx,
            total_chunks=len(chunks)
        )
        chunk_results.append(chunk_result)

    log_line(
        f"[CHUNKS_DONE] {endpoint_name} | {input_path.name} | "
        f"completed_chunks={len(chunk_results)}/{len(chunks)}"
    )

    return {
        "input": str(input_path),
        "status": "chunked_completed",
        "chunked": True,
        "chunk_count": len(chunk_results),
        "original_prompt_est": estimated_prompt_tokens,
        "min_chunks": min_chunks,
        "actual_chunks": len(chunks),
        "nodes": sum(r["nodes"] for r in chunk_results),
        "relationships": sum(r["relationships"] for r in chunk_results),
        "total_tokens": sum((r.get("total_tokens") or 0) for r in chunk_results),
        "chunk_results": chunk_results,
        "model": MODEL_NAME,
        "base_url": base_url,
        "endpoint_name": endpoint_name,
        "extracted_at": datetime.now().isoformat(),
    }


def pick_input_files() -> list[Path]:
    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {INPUT_DIR}")
    return files


def already_processed(input_path: Path) -> bool:
    base_output = _output_path_for_input(input_path, 0)
    chunked_output_1 = _output_path_for_input(input_path, 1)
    base_failed = _failed_output_path_for_input(input_path, 0)
    chunked_failed_1 = _failed_output_path_for_input(input_path, 1)

    return (
        base_output.exists()
        or chunked_output_1.exists()
        or base_failed.exists()
        or chunked_failed_1.exists()
    )


def endpoint_worker(base_url: str, file_queue: Queue, results: list, results_lock: threading.Lock):
    endpoint_name = _endpoint_label(base_url)

    while True:
        try:
            input_path = file_queue.get_nowait()
        except Empty:
            log_line(f"[WORKER_DONE] {endpoint_name} | no more files")
            break

        try:
            result = extract_from_file(input_path, base_url)

            with results_lock:
                results.append(result)

        except Exception as e:
            log_line(f"[ERROR] {endpoint_name} | {input_path.name} | {repr(e)}")

            with results_lock:
                results.append({
                    "input": str(input_path),
                    "status": "worker_error",
                    "chunked": False,
                    "nodes": 0,
                    "relationships": 0,
                    "total_tokens": 0,
                    "model": MODEL_NAME,
                    "base_url": base_url,
                    "endpoint_name": endpoint_name,
                    "extracted_at": datetime.now().isoformat(),
                })

        finally:
            file_queue.task_done()


def run():
    input_paths = pick_input_files()

    files_to_process = []
    skipped_count = 0

    for input_path in input_paths:
        if already_processed(input_path):
            log_line(f"[SKIP]  {input_path.name}")
            skipped_count += 1
            continue
        files_to_process.append(input_path)

    grand_total_nodes = 0
    grand_total_relationships = 0
    grand_total_tokens = 0

    log_line(f"[INFO]  Found {len(input_paths)} files")
    log_line(f"[INFO]  Processing {len(files_to_process)} files with {len(BASE_URLS)} dedicated endpoint worker(s)")
    log_line(f"[INFO]  Output dir: {OUTPUT_DIR.resolve()}")

    file_queue = Queue()
    for input_path in files_to_process:
        file_queue.put(input_path)

    results = []
    results_lock = threading.Lock()
    workers = []

    for base_url in BASE_URLS:
        t = threading.Thread(
            target=endpoint_worker,
            args=(base_url, file_queue, results, results_lock),
            daemon=False
        )
        t.start()
        workers.append(t)

    for t in workers:
        t.join()

    processed_count = 0
    for result in results:
        grand_total_nodes += result.get("nodes", 0)
        grand_total_relationships += result.get("relationships", 0)
        grand_total_tokens += result.get("total_tokens") or 0

        if result.get("status") != "worker_error":
            processed_count += 1

    log_line("===== ALL FILES DONE =====")
    log_line(f"Files found:         {len(input_paths)}")
    log_line(f"Files processed:     {processed_count}")
    log_line(f"Files skipped:       {skipped_count}")
    log_line(f"Grand total nodes:   {grand_total_nodes}")
    log_line(f"Grand total rels:    {grand_total_relationships}")
    log_line(f"Grand total tokens:  {grand_total_tokens}")


if __name__ == "__main__":
    run()
