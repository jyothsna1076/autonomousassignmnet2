#  Academic Helpdesk Agent

A multi-agent RAG assistant for IIT Jodhpur students, built with **Google ADK**, **Ollama (Qwen3 8B / Llama 3.1 8B)**, **LiteLLM**, **nomic-embed-text** and **ChromaDB**.

It answers questions about academic rules, curriculum, course slots and timetables, and writes (but **never sends**) e-mail drafts to course instructors. If a question is not covered by the documents, the agent says it doesn't know and logs the question for a human to answer. Every answer taken from the documents names its source (file name and page).

> Course project: *Autonomous Systems – Assignment 2*
> Authors: **Jyothsna**, **Pragna**, **Sriya** , **Vamsi**
> Indian Institute of Technology Jodhpur · 30 October 2026

---

##  Features

- **Document Q&A with citations**: policy, regulations, calendar and curriculum questions answered from 21 PDFs, with file name and page number.
- **Structured lookups**: course slots, lecture timings, course counts and instructor contacts from CSV tables.
- **E-mail drafting**: generates a draft saved to `drafts/`. Nothing is ever sent.
- **Safe refusal and logging**: out-of-scope questions get a fixed "I don't know" reply and are saved to `logs/unanswered_questions.csv`. The helpdesk logs automatically even if the model forgets to call the tool.
- **Two architectures**: single-agent (all tools) and multi-agent (orchestrator → policy / schedule / e-mail agents).
- **Model swap**: change the LLM with one argument (Qwen3 8B ↔ Llama 3.1 8B).
- **Evaluation harness**: 32 test questions × 3 runs, with accuracy, citation, groundedness, refusal and stability metrics.
- **Gradio chat UI** for Colab.

---

##  System design

### Tools

| Tool | What it does |
|---|---|
| `search_documents` | Finds the most relevant chunks in ChromaDB and returns them with file name and page. |
| `lookup_timetable` | Gives the slot and lecture timing of a course from `timetable.csv`. |
| `list_courses` | Counts or lists courses by department, level and type from `courses.csv`. |
| `get_instructor_contact` | Gives the instructor's name and e-mail from `faculty.csv`. |
| `draft_email` | Writes an e-mail draft and saves it in `drafts/`. It is never sent. |
| `log_unanswered` | Saves a question the agent cannot answer in `logs/unanswered_questions.csv`. |

### Agents

```
                     Helpdesk orchestrator
                    /          |            \
             Policy agent  Schedule agent  E-mail agent
         search_documents  lookup_timetable  get_instructor_contact
                           list_courses      draft_email

   Unknown question → standard refusal + log_unanswered
```

The single-agent version gives all six tools to one agent.

### Data

34 files in total: 21 PDFs (academic regulations, attendance and examination circulars, academic calendar, programme list, B.Tech./B.S./M.Sc. curriculum booklets) and 13 CSV sheets (department course lists and the slot system). They are split into **3,171 chunks** stored in ChromaDB with file name and page metadata. `prepare_csvs.py` merges the department sheets into `timetable.csv`, `courses.csv` and `faculty.csv`.

---

##  Repository structure

```
helpdesk-agent/
├── data/
│   ├── docs/                 # PDF documents (regulations, circulars, curricula, ...)
│   └── csv/                  # Department course sheets and slot system
├── prepare_csvs.py           # Builds timetable.csv, courses.csv, faculty.csv
├── ingest.py                 # Chunks documents and builds the ChromaDB vector store
├── tools.py                  # The six tools
├── helpdesk.py               # Helpdesk class: single / multi-agent setups (Google ADK)
├── evaluate.py               # Evaluation script
├── questions.csv             # 32 test questions with expected agent, source, keywords
├── requirements.txt
├── Helpdesk_Colab.ipynb      # Colab notebook that runs everything
├── drafts/                   # Generated e-mail drafts (created at runtime)
├── logs/                     # unanswered_questions.csv (created at runtime)
└── results/                  # Evaluation output (created at runtime)
```

---

##  Getting started (Google Colab)

The project is designed to run on a **Google Colab T4 GPU**.

1. Put the project folder in Google Drive at `MyDrive/helpdesk-agent/`, with documents in `data/docs/` and CSVs in `data/csv/`.
2. Open `Helpdesk_Colab.ipynb` in Colab and set **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom:

| Step | Cell | Notes |
|---|---|---|
| 1 | Check the GPU | `nvidia-smi` |
| 2 | Mount Drive, set `HELPDESK_DIR` | Change `PROJECT` if the folder is elsewhere |
| 3 | `pip install -r requirements.txt` | |
| 4 | Install Ollama | |
| 5 | Start the Ollama server | 8K context so tool schemas and passages fit |
| 6 | `ollama pull qwen3:8b` and `nomic-embed-text` | ~5 GB |
| 6b | `python prepare_csvs.py` | Re-run whenever a sheet changes |
| 7 | `python ingest.py` | Re-run whenever documents change |

Every new Colab session needs steps 1–7 again (about 3–5 minutes). After that, any later cell can be used.

### Using the agents

```python
from helpdesk import Helpdesk

# Single agent
single = Helpdesk('single')
r = await single.ask('What is the attendance requirement to sit in the Major exam?')
print(r['answer']); print(r['agents'], r['tools'], r['seconds'], 's')

# Multi-agent
multi = Helpdesk('multi')
r = await multi.ask('What is the slot for the course Autonomous Systems?')

# Swap the model
alt = Helpdesk('multi', model='ollama_chat/llama3.1:8b')
```

`await` works directly in Colab cells. Each call without a `session_id` uses a fresh session.

### Gradio chat UI

The last cells of the notebook launch a Gradio app where you can choose single or multi-agent mode and the language model, see the route and tools used, and start a new conversation. A simpler `gr.ChatInterface` version with `share=True` is also included.

### Inspecting logs and drafts

```python
import pandas as pd
pd.read_csv('logs/unanswered_questions.csv')   # refused / unanswered questions
print(open(sorted(glob.glob('drafts/*.txt'))[-1]).read())   # latest e-mail draft
```

---

##  Evaluation

32 questions in `questions.csv`, each with the expected agent, expected source file and keywords the answer must contain.

| Type | Example | Count |
|---|---|---|
| Policy (P) | What is the attendance requirement to sit in the Major exam? | 13 |
| Schedule (S) | What is the slot for the course Autonomous Systems? | 7 |
| E-mail (E) | Write an email to the instructor of CSL7650 that I need leave for 2 days. | 4 |
| Not in our data (N) | Slot of "Quantum Basket Weaving"; e-mail to instructor of CSL7570 | 2 |
| Out of scope (O) | What are the available dining options on the campus? | 6 |

Run the evaluation (each question 3 times, fresh session each time):

```bash
# 1. Multi-agent, Qwen3 8B
python evaluate.py --mode multi --runs 3

# 2. Model swap: multi-agent, Llama 3.1 8B
ollama pull llama3.1:8b
python evaluate.py --mode multi --runs 3 --model ollama_chat/llama3.1:8b

# 3. Single-agent baseline, Qwen3 8B
python evaluate.py --mode single --runs 3
```

A full run takes roughly 15–30 minutes on a T4. Results are written to `results/` (`model_comparison.csv` and `per_question_*.csv`).

**Metrics:** accuracy, citation present, groundedness (cited source is the expected file), correct refusal, false refusal, routing accuracy, and changes across runs.

---

##  Results

Accuracy, citation and groundedness are averaged over the 24 answerable questions (P, S, E). Each question was run 3 times.

| Metric | Multi, Qwen3 | Multi, Llama 3.1 | Single, Qwen3 |
|---|---|---|---|
| Accuracy (%) | 63.9 | 50.0 | **75.0** |
| Citation present (%) | 80.6 | 62.5 | **87.5** |
| Groundedness (%) | 61.1 | 59.7 | **79.2** |
| Out-of-scope refused (%) | 100 | 100 | 100 |
| Not-in-data refused (%) | 50.0 | 100 | 100 |
| False refusals (%) | 19.4 | 33.3 | **12.5** |
| Refusals logged (%) | 100 | 100 | 100 |
| Routing accuracy (%) | 96.2 | 92.3 | n/a |
| Questions with different answers across 3 runs | 7 | 13 | 13 |
| Questions whose correctness changed | 1 | 6 | **0** |
| Avg. time per question (s) | 3.7 | 4.1 | 3.8 |

**Accuracy by question type (%)**

| Type | Multi, Qwen3 | Multi, Llama 3.1 | Single, Qwen3 |
|---|---|---|---|
| Policy (13) | 53.8 | 59.0 | 53.8 |
| Schedule (7) | 76.2 | 47.6 | 100.0 |
| E-mail (4) | 75.0 | 25.0 | 100.0 |

### Key findings

1. **Out-of-scope questions are handled well.** All six were refused and logged in every run of every set-up.
2. **The single agent did best.** Qwen3 single-agent reached 75.0% accuracy vs. 63.9% for multi-agent, with fewer false refusals. The extra routing step is one more place to go wrong.
3. **Qwen3 often skips the search tools.** In multi-agent Qwen3, 25 of 72 answers to answerable questions used no search or lookup: 14 wrong refusals and 11 answers from memory.
4. **Answers from memory can look right but be made up.** For example, the model cited a non-existent `academic_regulations.pdf`, and invented a slot, days and room for one course, citing a page that doesn't exist. This is why groundedness is reported separately from accuracy.
5. **Qwen3 is better overall; Llama 3.1 reads documents well.** Llama 3.1 is slightly better on policy questions (59.0% vs. 53.8%) but refuses a third of answerable questions and is less stable.
6. **Some questions failed in every set-up** (P07, P08, P09), mostly because the needed date or rule wasn't found or didn't match the expected answer.
7. **Not-in-data questions.** Multi-agent Qwen3 answered "not available in the records" for one fake course instead of using the standard refusal, so it was not logged and counted as a failure.

See the full report (`autonomousassignment2report.pdf`) for per-question results and the table of answers that changed across runs.

---

##  Limitations

- Qwen3 8B sometimes refuses or answers **without searching first**, causing false refusals and occasional made-up sources.
- Small local models give **different answers across runs**; e-mail drafts always vary in wording.
- Answers depend entirely on the indexed documents. **AI-generated responses may be incorrect, so always check the cited academic documents** for authoritative rules.
- E-mails are **drafts only**; the system has no sending capability by design.

---

##  Tech stack

- [Google ADK](https://google.github.io/adk-docs/) (agent framework)
- [Ollama](https://ollama.com/) + [LiteLLM](https://github.com/BerriAI/litellm) (local LLM serving)
- Qwen3 8B, Llama 3.1 8B (language models)
- nomic-embed-text (embeddings)
- [ChromaDB](https://www.trychroma.com/) (vector store)
- [Gradio](https://www.gradio.app/) (chat UI)
- Google Colab T4 GPU

---

##  Report

The full project report is in [`autonomousassignment2report.pdf`](autonomousassignment2report.pdf).
