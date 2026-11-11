# AI Study Assistant

A small chat program with one **Orchestrator LLM** as the main controller.
The Orchestrator reads your message, splits it into separate questions, and
sends each question to one of three specialist LLMs (Physics, Biology or
General Knowledge), to the local knowledge base (RAG), or to one of three
tools (Calculator, Weather, Web Search).

For example:

> "What is Newton's first law of motion and who is the current PM of Pakistan?"

is split into a **physics** task and a **web_search** task. The answers are
then combined into one reply.

Everything is written with plain Python dictionaries and simple `if` / `elif`
statements - no pydantic and no agent frameworks.

## The files

| File | What it does |
|------|--------------|
| `main.py` | The chat loop. It reads your question, calls the agent, and prints the answer. It also handles `attach`, `documents` and `remove`. |
| `agent.py` | The brain. It contains the Orchestrator, the specialist LLMs, the RAG route, and the tool loop. |
| `tools.py` | The three tools: `calculator`, `get_weather` and `web_search`. |
| `rag.py` | The local knowledge base. It reads documents, builds a TF-IDF index, retrieves the most relevant chunks, and runs OCR on scanned PDFs when needed. |
| `knowledge/` | Your own study files. The `knowledge` route answers from these documents. |
| `.rag_cache/` | Created automatically. Stores OCR text so scanned PDFs are only processed once. |
| `requirements.txt` | The packages the program needs. |

## Setting it up

1. Install the packages:

   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file next to `main.py`:

   ```
   LLM_API_KEY=your_openrouter_key
   LLM_MODEL=minimax/minimax-m3.1-flash
   ORCHESTRATOR_MODEL=dots-studio/dots-3-note-preview:free
   PHYSICS_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
   GENERAL_MODEL=minimax/minimax-m3.1-flash
   ```

   `ORCHESTRATOR_MODEL`, `PHYSICS_MODEL` and `GENERAL_MODEL` are optional.
   If one is missing, the program uses `LLM_MODEL` instead.

   The example above uses specific free models. Do **not** use
   `openrouter/free` for this project: it is a random router, and it sometimes
   picks a content-safety model that answers `"User Safety: safe"` instead of
   answering your question.

3. Run the program:

   ```
   python main.py
   ```

4. Type `exit` to stop.

## Adding documents

There are two ways to add material to the knowledge base:

- Put `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json` or `.html` files in the
  `knowledge/` folder.
- Or use the `attach` command in the chat:

  ```
  attach "C:\Users\me\Documents\Physics Notes.pdf"
  ```

  The file is copied into `knowledge/` and indexed.

Other commands:

| Command | What it does |
|---------|--------------|
| `documents` | Lists the indexed files and their chunk counts. |
| `remove "filename"` | Deletes a file from `knowledge/` and the index. |
| `exit` | Stops the program. |

## How a message is handled

`run_agent()` in `agent.py` does the whole job in five steps:

1. **Plan.** `plan_tasks()` asks the **Orchestrator LLM** to split the message
   into one or more tasks and choose a route for each task. The Orchestrator
   replies with JSON like this:

   ```json
   {
       "tasks": [
           {"question": "What is Newton's first law of motion?", "route": "physics"},
           {"question": "Who is the current Prime Minister of Pakistan?", "route": "web_search"}
       ]
   }
   ```

   These are the possible routes:

   | Route | Who answers |
   |-------|-------------|
   | `physics` | The Physics LLM |
   | `biology` | The Biology LLM |
   | `general` | The General Knowledge LLM |
   | `knowledge` | The local knowledge base (RAG) |
   | `calculator` | The `calculate` tool |
   | `weather` | The `get_weather_tool` tool |
   | `web_search` | The `web_search_tool` tool |

2. **Check the knowledge base.** `apply_knowledge_fallback()` checks the local
   documents. A task is switched to the `knowledge` route when it mentions
   "my notes", "my documents", "my PDF", a file name, or when enough of the
   question's words appear in the indexed documents.

3. **Answer.** `answer_task()` uses a simple `if` / `elif` to send each task to
   its chosen worker. The Physics, Biology and General LLMs answer with text.
   For a tool route, `run_tool_loop()` lets the General LLM call that one tool
   and then explain the result.

4. **Combine.** `combine_results()` puts the sub-answers into one reply. Each
   answer is numbered under its question. The final `topic` and `route` fields
   list every worker that was used, and the `used_*` flags are combined too.

5. **Remember.** Your message and the combined answer are added to
   `message_history`, so the next question can use them as context.

`main.py` prints `Route: <route>` and `Sources: <files>` so you can see which
workers were used and which documents were read.

## The tools

| Tool | Used for |
|------|----------|
| `calculate` | Simple maths with two numbers. |
| `get_weather_tool` | Current weather for one city (Open-Meteo). |
| `web_search_tool` | Facts that change over time (OpenRouter web search). |

The tools themselves in `tools.py` are unchanged from the original version.

## The knowledge base (RAG)

The `knowledge` route adds retrieval-augmented generation to the assistant.
It works like this:

1. **Load.** `rag.py` reads every supported file inside `knowledge/`.
   Sub-folders are allowed. PDF text is read with `pypdf`; pages without a
   text layer are rendered with `PyMuPDF` and read with `RapidOCR`.
2. **Chunk.** Each document is split into chunks of about 800 characters,
   with 150 characters of overlap.
3. **Index.** A TF-IDF index is built in memory. This is plain Python, so
   no vector database is needed.
4. **Retrieve.** When a question is routed to `knowledge`, the chunks with
   the highest cosine similarity are selected (four by default). File names
   are checked too, so "summarize Goals.pdf" finds that file. If the question
   names a file, only that file is searched.
5. **Answer.** The chunks are added to the LLM prompt as `PROVIDED CONTEXT`.
   The model is told to answer only from that context and to say so when the
   answer is not there.

Follow-up questions such as "which sentence comes after this one" reuse
the previous user message when building the search query.

To add your own material, use `attach` or put the files in `knowledge/`.
The file names are printed as `Sources:` after the answer. PDF answers also
include `[Page N]` markers in the context, so the model can cite the page.

**PDF notes:** PDFs with selectable text are read directly. Scanned or
image-only PDFs are read with OCR. To keep startup fast, OCR runs only when
you mention the scanned file by name, for example
*"In English.pdf, what does page 3 say?"*. The first run takes a few
minutes; the text is then stored in `.rag_cache/`, so later runs are fast.
Delete `.rag_cache/` if you want to force the OCR to run again.

You can turn OCR off by setting `OCR_ENABLED = False` at the top of
`rag.py`.

The retrieval settings are at the top of `rag.py`:
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SCORE`, `OCR_ENABLED`,
and `OCR_DPI`.

## The settings

You can change these at the top of `agent.py`:

| Setting | What it does |
|---------|--------------|
| `MAX_TOOL_ROUNDS` | How many times the model may ask for a tool (default `5`). |
| `ORCHESTRATOR_INSTRUCTIONS` | The system prompt that tells the Orchestrator how to split and route. |
| `PHYSICS_INSTRUCTIONS` / `BIOLOGY_INSTRUCTIONS` / `GENERAL_INSTRUCTIONS` | The system prompt given to each specialist. |
| `KNOWLEDGE_INSTRUCTIONS` | The system prompt for the RAG route. It tells the model to use only the retrieved context. |
| `KNOWLEDGE_ROUTE_MIN_SCORE` | The similarity score needed to switch an ordinary study question to `knowledge` (default `0.10`). |
| `KNOWLEDGE_ROUTE_MIN_COVERAGE` | The share of the question's keywords that must appear in the document (default `0.6`). |

## How the answers are written

Every worker is asked for a **short, direct answer** - one or two plain
sentences, with no headings, no lists, and no extra background. The web search
tool is asked for a single sentence that states the fact directly, for example
*"The current Prime Minister of Pakistan is Shehbaz Sharif."*

The rules live in `SHORT_ANSWER_RULES` at the top of `agent.py`, and they are
added to the Physics, Biology, General, Knowledge, and Tool prompts.

## The answer format

Every specialist and tool answer must be JSON with these keys:

```json
{
    "answer": "The answer text",
    "topic": "Physics",
    "difficulty": "Beginner",
    "used_calculator": false,
    "used_weather": false,
    "used_web_search": false,
    "used_knowledge": false
}
```

`parse_json_response()` reads that JSON even if the model wraps it in
``` ```json ``` blocks. If the model does not return JSON, the raw text is used
as the answer, so the program still works.

For a single task, the result is returned unchanged. For several tasks,
`combine_results()` creates the final `answer`, `topic`, `difficulty`,
`used_*` flags, `sources`, and `route`.

## If a model fails

If the LLM request fails, the error is shown by `main.py`. Check your API key
and model names in `.env`. If a free model is permanently unavailable, change
that role's model to another free model that supports tools.
"# RAG" 
