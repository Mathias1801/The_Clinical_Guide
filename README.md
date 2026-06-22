# Demo

Find Conceptual Demo hosted via streamlit here: https://kg-rag-demo-8vjfpvifgemxgmvtocmly3.streamlit.app/ 

(The demo uses mock data. Demo may be outdated)


# Repository Information

Data folders in this repository hold the following information:
- `data/logs` holds log database `"log.db"` of live runs made in `"streamlit_app.py"`
- `data/patient_notes` holds data for testing with `scripts/batch_testing.py`.


Scripts folder contains:
- `analyze_batch_outputs.py` inspection of test data
- `batch_testing.py` main pipeline script, setup for running full datasets
- `extraction.py` extraction of relationship triplets from scrape files
- `judge_utils.py` judge configuration script
- `scrape.py` script used to scrape relevant files from sundhed.dk
- `vector_search_transformer.py` script for creating vector embeddings of diagnosis nodes

The root folder holds:
- `requirements.txt` needed to run for installation of packages.
- `streamlit_app.py` the live streamlit version of pipeline.
- and other standard github elements.

# Reproduce our results with the following setup:

We reached out to relevant actors to get a permission agreement. Get relevant permissions if needed.

### Get Data Foundation:
- `scripts/scrape.py` to scrape files.
- `scripts/extraction.py` to extract relationship triplets.
- `scripts/vector_search_transformer` to create vector embeddings. This uses the model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

### Pipeline Setup:
- `scripts/judge_utils.py` holds judge configuration
- `scripts/batch_testing.py` runs xlsx data located in `data/patient_notes` file through the entire pipeline.
- `scripts/analyze_batch_outputs.py` allows inspection of run files from batch testing.
- The model used throughout is `gemma3:27b`

### Live Testing:
- `streamlit_app.py` live version of the application. Uses only the Weighted-KG variant unless changed.

### Setup `.env` file with the following for direct application:

```env
BASE_URL1 =
BASE_URL2 =
BASE_URL3 =
ENDPOINT =
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=
```

If not applicable to your case, change scripts accordingly.

# Overview of main pipeline:

`scripts/batch_testing.py`

## Batch Clinical KG Diagnosis Pipeline

This script runs a batch pipeline over Excel files containing patient notes and expected diagnoses. It extracts clinical retrieval terms, retrieves diagnosis candidates from vector indexes, expands related knowledge graph relations through Neo4j, reranks candidates with a local LLM, generates diagnosis outputs, and optionally evaluates the results with judge functions.

### What the script does

For each row in the input Excel files, the pipeline:

1. Reads a patient note from the `pso_note` column.
2. Reads the expected diagnosis from the `diagnosis` column.
3. Uses a local LLM to extract structured Danish clinical retrieval terms.
4. Searches diagnosis-like nodes using hybrid dense + lexical retrieval.
5. Reranks diagnosis candidates with the local LLM.
6. Expands selected diagnosis nodes into nearby Neo4j knowledge graph relations.
7. Reranks retrieved KG relations with the local LLM.
8. Builds one or more diagnostic prompts:
   - with KG context
   - without KG context
   - with KG context weighted as main evidence
9. Generates diagnostic answers.
10. Optionally runs judge evaluations.
11. Saves one JSON output per processed row.
12. Saves a manifest file summarizing the batch run.

