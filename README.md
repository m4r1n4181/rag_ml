# ML Course Assistant: RAG in Serbian, with a real evaluation

A study assistant for the university course *Mašinsko učenje 1* (Machine Learning 1).
Students ask questions in Serbian and get answers built **only** from the lecture slides and the
course textbook, with every source cited and the original slide shown next to the answer.

**Live demo:** _add link after deploying_

What makes this more than a "chat with PDF" project is the evaluation. Every design choice was
tested on 72 official exam questions, decided on a dev split and reported on a held-out test split.

| Held-out test set (48 questions) | Before | After |
|---|---|---|
| Relevant material retrieved (Recall@10) | 0.42 | **0.59** |
| First relevant slide ranked first (MRR) | 0.71 | **0.94** |
| Answer completeness (LLM judge) | 0.46 | **0.65** |
| Answer faithfulness to sources (LLM judge) | 0.81 | **0.87** |

---

## Try the demo

Click one of the example questions, or ask your own (in Serbian or English). Good ones to try:

| Question | What to look at |
|---|---|
| *Objasniti pojam regularizacije u kontekstu linearne regresije i na koje načine se ona primenjuje.* | The question that started the project: the original version answered "not in the materials". Now the sources panel shows slides 25–30 of the linear regression lecture (Ridge, Lasso). |
| Then: *A po čemu se Lasso razlikuje od Ridge?* | A follow-up question. It is rewritten into a standalone question before searching. |
| *Objasniti ulogu parametra brzine učenja kod gradijentnog silaska.* | Biggest single improvement in the evaluation: completeness 0.12 → 0.88. |
| *Kako se vrši izbor modela kod stabla odluke ukoliko se koristi orezivanje?* | Completeness 0.14 → 0.64 after the changes. |
| *Kako radi transformer arhitektura?* | Not covered by the course. The assistant should say it can't find it in the materials instead of answering from general knowledge. |

On the right, every answer lists its sources (lecture + slide, or textbook page). For slides you
can see the original image, which matters for formulas that don't survive PDF text extraction.

The public demo is limited to 20 questions per session to keep API costs bounded.

---

## How it works

```
Indexing (once)
  PDFs ─► text extraction + cleanup ─► chunks ─► "lecture | slide title" header ─► e5 embeddings ─► ChromaDB

Answering
  question ─► rewrite if it's a follow-up ─► vector search, top 10 ─► Claude Haiku ─► answer + cited sources
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | `intfloat/multilingual-e5-large-instruct` | Trained for retrieval and multilingual. Beat the original MiniLM on every metric: Recall@10 0.42 → 0.54, MRR 0.71 → 0.91. ~0.6 s per query on CPU, fine for chat. |
| Chunking | One slide = one chunk; textbook in 800-char chunks with 150 overlap | Slides are short and self-contained; textbook pages are long prose. |
| Context header | Every chunk starts with `Predavanje: <lecture> \| Slajd: <title>` | A slide titled "Lasso" never says "regularization" or "linear regression". The header adds that context. MRR 0.91 → 0.94, Recall@10 0.54 → 0.59. |
| Vector store | ChromaDB (persistent, cosine) | Simple, local, enough for ~800 chunks. |
| Keyword search | BM25 with a small Serbian stemmer, fused with Reciprocal Rank Fusion | Implemented and measured, **off by default**: it helps the small MiniLM model a lot (Recall 0.42 → 0.51) but makes no meaningful difference with e5. |
| Reranker | `bge-reranker-v2-m3` cross-encoder | Implemented, off, not evaluated: slow on CPU, and MRR is already 0.94. The remaining gap is recall, which reranking can't fix. |
| Chunks per answer | 10 | Chosen on the dev set by answer completeness: 0.61 / 0.65 / 0.68 for 5 / 8 / 10 chunks. |
| LLM | Claude Haiku 4.5, streamed | Cheap and fast; answers must stay within the retrieved context. |
| UI | Streamlit | Chat on the left, sources and slide images on the right. |

---

## Evaluation

### Data

- **72 official exam questions** of the course, each with a reference answer. The reference
  answers were written with an LLM as a study aid. A manual check of 178 of their key points
  found **98% stated in the course materials**, so they are a fair target.
- Split **24 dev / 48 test**, stratified by exam section. All decisions were made on dev; test was
  used once for the final numbers.
- The reference answers are not in the index. The assistant only sees slides and the textbook.

### Relevance labels

Retrieval metrics need to know which chunks are relevant to each question. Labeling ~800 chunks
× 72 questions by hand isn't realistic, so labels were built by **pooling** (the method used in
TREC search evaluations): the top 10 chunks from each of 8 configurations were pooled per question
(~28 chunks), shuffled so the grader can't tell which system found what, and graded 0/1/2 by an
LLM (Claude Sonnet 5.5) using the reference answer as a guide. 2,062 judgments, spot-checked by hand.

### Metrics

| Metric | Question it answers |
|---|---|
| **Recall@10** | Of all chunks useful for this question, what share is in the top 10? (main metric: the LLM can ignore noise, but can't use a slide it never got) |
| **Precision@10** | What share of the top 10 is useful? |
| **MRR** | How high is the first useful chunk? (1 = always first) |
| **nDCG@10** | Ranking quality that also weighs "very useful" above "partly useful" |
| **Completeness** | The reference answer is split into 3–8 key points; an LLM judge checks which ones the answer covers (yes = 1, partial = 0.5) |
| **Faithfulness** | Share of statements in the answer that are supported by the retrieved sources |

### Results: retrieval ablation (test set, top 10)

| Configuration | Recall | Precision | MRR | nDCG |
|---|---|---|---|---|
| MiniLM, vector search (original setup) | 0.420 | 0.402 | 0.712 | 0.418 |
| MiniLM + BM25 | 0.505 | 0.494 | 0.823 | 0.533 |
| MiniLM + headers | 0.432 | 0.419 | 0.760 | 0.462 |
| MiniLM + headers + BM25 | 0.510 | 0.498 | 0.837 | 0.573 |
| e5 | 0.543 | 0.537 | 0.906 | 0.625 |
| e5 + BM25 | 0.589 | 0.579 | 0.905 | 0.655 |
| **e5 + headers (shipped)** | **0.591** | **0.583** | **0.937** | **0.691** |
| e5 + headers + BM25 | 0.609 | 0.594 | 0.936 | 0.688 |

Dev-set tables and per-k results: [data/eval/results.md](data/eval/results.md).

### Results: answers (test set)

| Pipeline | Retrieval | Chunks | Completeness | Faithfulness |
|---|---|---|---|---|
| Original (index and prompt restored from git) | MiniLM | 5 | 0.46 | 0.81 |
| Current | e5 + headers | 10 | **0.65** | **0.87** |

### Can the LLM judge be trusted?

The judge was compared with a human on 15 answers (103 key points): **69% agreement, Cohen's kappa
0.53**. The disagreements are not random: in 27 of 32 the judge is *stricter* than the human, and in
the cases checked it was usually right (for example, it caught a wrong index in a formula). So
absolute completeness is conservative, and before/after comparisons are fair because the same
judge grades both.

### What the evaluation changed

- **A crude first metric gave a wrong conclusion.** Counting a hit whenever a chunk came from the
  right lecture rewarded the lecture's table-of-contents slide and suggested BM25 *hurts* e5.
  Slide-level labels showed it doesn't. That is why the evaluation was rebuilt.
- **Broken text in the textbook.** While checking labels, the textbook (a LaTeX PDF) turned out to
  have ~4,300 broken characters: "č" extracted as "ˇc", "ć" as "´c", "fi" as a ligature glyph,
  "između" as "izmedu". Fixed in `pdf_loader.py` and covered by tests. Slides also carried a
  footer and PowerPoint bullet characters on every page.
- **A bug in the evaluation itself.** The first answer judge saw the retrieved text without the
  source labels, so every "(slajd 26)" citation was flagged as made up. The judge now sees exactly
  the prompt the model saw.
- **A hypothesis that was ruled out.** Because the reference answers are LLM-generated, the
  suspicion was that they ask for things the course never covers. 98% of their key points are in
  the materials, so low completeness comes from retrieval and the answers, not the reference.

### Limitations

- Recall@10 is 0.59: about 40% of useful material still doesn't reach the model. Next steps:
  split multi-part exam questions into sub-queries, and add neighboring slides of each hit.
- Formulas are extracted as broken text; the model reads text, not slide images. A vision model
  could describe each slide at indexing time.
- Relevance labels and answer grades come from an LLM, checked against a human on a sample.
- 48 test questions is a small sample; differences of a few points are within noise.
- Query rewriting for follow-ups is not evaluated (all exam questions are standalone).

---

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                   # add ANTHROPIC_API_KEY
streamlit run ui/app.py
```

The index (`data/chroma_db`) is included. Rebuild it with `python build_db.py` only if you change
the embedding model, chunking or the PDFs (PDFs go in `data/raw/`, not included).

Check retrieval for any question, without API calls:

```bash
python debug_retrieval.py "Objasniti regularizaciju kod linearne regresije"
```

## Deploy (Hugging Face Spaces, free CPU)

1. Create a Space with the **Docker** SDK.
2. In the Space settings, add a secret `ANTHROPIC_API_KEY`.
3. Push this repository to the Space. The Space's `README.md` must start with:
   ```yaml
   ---
   title: ML Course Assistant
   sdk: docker
   app_port: 7860
   ---
   ```
   Hugging Face only accepts binary files (slide images, the index) through Git LFS:
   `git lfs track "*.png" "*.sqlite3" "*.bin"` before committing to the Space.

The `Dockerfile` installs CPU-only PyTorch and downloads the embedding model at build time. The
app needs ~3 GB of RAM. Set a monthly spend limit in the Anthropic console; the per-session cap
(`MAX_QUESTIONS_PER_SESSION`) only slows down abuse.

## Tests and evaluation

```bash
pip install -r requirements-dev.txt
pytest                                          # unit tests, also run by GitHub Actions

python -m evals.retrieval --subset test         # retrieval metrics, no API cost
python -m evals.answers --subset test           # answer quality (API calls, cached)
python -m evals.human_check --n 15              # grade answers yourself, compare with the judge
python -m evals.report                          # rebuild data/eval/results.md
```

Reproducing the ablation from scratch needs the PDFs and the ablation indexes; the commands are
at the top of `evals/systems.py`. All LLM calls are cached in `data/eval/`, so rerunning
the metrics costs nothing. LLM steps stop automatically above `MAX_EVAL_SPEND_USD` (default $15).

## Project structure

```
src/
  ingestion/   pdf_loader.py (parsing + text cleanup), chunker.py, ingest.py
  retrieval/   vector_store.py (embeddings, Chroma, search, RRF, reranker), lexical.py (BM25)
  generation/  prompts.py, llm.py, rag_chain.py (query rewriting + RAG)
ui/app.py      Streamlit app
evals/         runs, relevance labeling, retrieval and answer metrics, human check, report
tests/         unit tests
data/eval/     questions, dev/test split, labels, cached runs and answers, results
```

The course materials (slides, textbook) belong to the course authors and are used here for a
non-commercial study project.
