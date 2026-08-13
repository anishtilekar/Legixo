[![CI](https://github.com/anishtilekar/Legixo/actions/workflows/ci.yml/badge.svg)](https://github.com/anishtilekar/Legixo/actions/workflows/ci.yml)

# Legixo Q&A API

This is my take-home project for the Legixo Gen AI internship. It's a small API that
answers questions about a set of documents. It only answers using the documents — it
never makes things up — and it always tells you which document/chunk the answer came
from. If the documents don't have the answer, it says so instead of guessing.

It's built with **Python**, **LangGraph** (for the question-answering flow),
**Pinecone** (to store and search the documents), and **Together.ai** (for the AI model
and embeddings).

> Everything in `corpus/` is made-up/fictional — fake companies, fake court cases, fake
> numbers. No real data anywhere.

---

## What's in this repo, quickly

- `app/` — the actual code (FastAPI server + the LangGraph flow)
- `corpus/` — the documents it answers questions about
- `eval/` — my test questions and how the app did on them
- `docs/` — extra notes (how the graph works, my build log)
- `tests/` — automated tests that don't need any API keys to run

---

## How to run this yourself

### 1. Requirements

- **Python 3.11** (not 3.12+, and not 3.14 either — a couple of the packages this
  project uses don't support those versions yet).
- A free [Together.ai](https://api.together.xyz) account — for the AI model and the
  embeddings.
- A free [Pinecone](https://app.pinecone.io) account — for storing the documents as
  vectors so they can be searched.

### 2. Install

```bash
git clone https://github.com/anishtilekar/Legixo.git
cd Legixo
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API keys

```bash
copy .env.example .env
```

Open `.env` and paste in your own `TOGETHER_API_KEY` and `PINECONE_API_KEY`. That's the
only thing you need to change. `.env` is in `.gitignore` so it never gets committed —
only `.env.example` (with fake placeholder values) is in the repo.

If you forget to fill them in, the app tells you exactly which key is missing instead
of just crashing.

### 4. Load the documents into Pinecone

```bash
python -m scripts.ingest --reset
```

This reads every file in `corpus/`, splits them into small chunks, turns each chunk
into a vector (an embedding), and uploads them to Pinecone. You don't need to create
the Pinecone index yourself — the code creates it automatically the first time you run
this, with the right settings (1024 dimensions, cosine similarity).

You should see something like:

```
files processed:  30
chunks upserted:  93
stale pruned:     0
```

**What if I run this command twice?** Nothing bad happens. Each chunk gets an ID based
on its file name and position, so running it again just overwrites the same chunks
instead of creating duplicates. I tested this — running it twice in a row still shows
93 chunks both times, not 186.

### 5. Start the server

```bash
python -m uvicorn app.main:app --port 8000
```

Now open **http://127.0.0.1:8000/** in a browser — there's a small page I built where
you can type a question and see the answer, the sources it used, and even a trace of
every step the AI went through to get there. There's also
**http://127.0.0.1:8000/docs**, which is an auto-generated page for testing the API
directly.

Note: you can only ask questions through the API (or that web page, which just calls
the API) — there's no separate command-line tool for asking questions, only for
loading the documents. That's on purpose, per the assignment.

### 6. Try asking something

A question it can answer:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is the notice period in the Bluecrest employment agreement?\"}"
```

```json
{
  "answer": "The notice period is 60 days written notice [S2].",
  "status": "answered",
  "citations": [
    { "marker": "[S2]", "source_path": "corpus/02_employment_agreement_excerpt.md", "score": 0.8387 }
  ],
  "attempts": 1
}
```

A question it *can't* answer (this one's a trick question — it mixes up two unrelated
documents on purpose):

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is the notice period at Harbor Bean Roasters?\"}"
```

```json
{
  "answer": "I could not find this in the provided documents.",
  "status": "not_found",
  "citations": [],
  "attempts": 3
}
```

---

## How the question-answering part works

I used LangGraph so the process isn't just "one big prompt" — it's broken into small
steps, each one doing one job, and it can make a decision (a branch) about what to do
next.

```
question
   │
   ▼
normalize → search Pinecone → check the results are actually good enough
                                       │
                         ┌─────────good enough─────────┐
                         │                              ▼
                    not good enough              write the answer
                         │                              │
              (try again, up to 2 times)         check every citation is real
                         │                              │
                         ▼                     citation missing/fake? drop it
              still not good enough?                    │
                         │                    still have a real citation?
                         ▼                          │            │
                  say "can't find it"              yes           no
                                                     │            │
                                                     ▼            ▼
                                                  done      say "can't find it"
```

In plain terms, the 9 steps are:

1. **normalize_question** — clean up the input
2. **retrieve** — search Pinecone for chunks that might be relevant
3. **rerank** — (optional, off by default) re-sort the results with a second model
4. **grade_context** — an AI check: "do these chunks actually answer the question?" — **this is the branch**
5. **rewrite_query** — if not, try rephrasing the search and go back to step 2 (**this is the loop**, capped at 2 retries so it can't run forever)
6. **generate_answer** — if yes, write the answer using only the chunks found, with `[S1]`, `[S2]` style citations
7. **verify_citations** — double-check every citation actually points to a real chunk (if the AI made one up, that part of the answer gets removed)
8. **no_answer** — the "I don't know" path
9. **finalize** — wrap everything up and send the response

More detail (with a proper diagram) is in [`docs/langgraph.md`](docs/langgraph.md).

There are two safety limits so it can never get stuck: it will only retry searching
**2 times** before giving up, and there's a second hard limit on top of that in case
something ever goes wrong with the first one.

---

## About the documents

The assignment gave 6 sample documents. I used those (files `01` to `06`, unchanged)
and also wrote **24 more documents myself**, in the same made-up style, so I'd have
enough data to properly test the search and grading. With only 6 documents, the search
step would basically always return everything anyway, so it wasn't really testing
anything.

I tried to make the extra documents genuinely tricky — for example there are three
different fake companies each with a different notice period (60 / 30 / 90 days), so
the app actually has to pick the right document, not just the right topic.

If you'd rather test with just the original 6 files:

```bash
python -m scripts.ingest --path path/to/original_six --reset
```

---

## Testing

**Automated tests** (no API keys needed, run instantly):

```bash
pytest tests/ -q
```

**My own test questions** — 33 questions I wrote myself, checking that the right
document gets cited and that questions with no real answer get refused. Start the
server first, then:

```bash
python -m scripts.run_eval --repeat 3
```

- The questions themselves: [`eval/test_cases.json`](eval/test_cases.json)
- Results + my notes on what passed/failed: [`eval/results.md`](eval/results.md)

Current result: **33 out of 33 pass**, and every question that should be refused
correctly gets refused.

---

## Extra things I tried

The assignment mentioned reranking and hybrid search as optional bonus features. I
built both (you can turn them on with `RERANK_ENABLED=true` and
`RETRIEVAL_MODE=hybrid` in `.env`), but when I actually tested them side by side
against plain search, neither one made the results better — reranking was actually a
bit worse, and hybrid search was the same but slower. So I left them both off by
default and wrote up what I found in [`eval/ablation.md`](eval/ablation.md) instead of
just turning them on and hoping they helped.

I also added an optional LangSmith tracing hookup (set `LANGCHAIN_TRACING_V2=true` in
`.env`) if you want to see every step logged externally, and a small preflight check
that gives a clear error message if you forget to set your API keys properly.

---

## What I'd still improve / known issues

Being upfront about what's not perfect:

- Occasionally (very rarely) the AI grader is a bit inconsistent between runs, even
  though I set temperature to 0 to make it as consistent as possible. When this
  happens it only ever causes a **safe** mistake — refusing a question it could
  actually answer — never making something up.
- Only works with markdown files right now — no PDF or Word doc support.
- No conversation memory — every question is answered on its own, with no memory of
  earlier questions.
- Getting to 33/33 on my test questions took a few real bug fixes along the way (not
  just tweaking numbers) — details in [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) if
  you're curious.

---

## Cost

Very cheap — roughly $0.0003 per question. Running my entire 33-question test set
three times over costs about 3 cents. All my testing for this whole project came to
well under a dollar.
