# ⚡ Spark Backend

The RAG-powered FastAPI backend for Spark — an AI skill coach for corporate professionals in India.

## What It Does

Spark is an AI coaching assistant that lives inside a corporate dashboard. It answers workplace questions, drafts documents (performance reviews, emails, feedback messages), and personalises advice to each user's communication style via Mirror Mode.

The backend handles:
- **RAG pipeline** — searches a local knowledge base before answering
- **Document drafter** — detects intent and generates complete workplace documents
- **Mirror Mode** — analyses a user's writing style and personalises responses
- **Knowledge base management** — admin routes to add, upload, and delete documents

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI |
| LLM | Llama 3.1 8B via Groq API (free) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local, free) |
| Vector database | ChromaDB (local) |
| RAG orchestration | LangChain |

## Project Structure

```
backend/
├── main.py              # FastAPI server — all routes
├── rag_engine.py        # RAG pipeline — load, embed, search, answer
├── requirements.txt     # Python dependencies
├── Procfile             # Deployment start command
├── runtime.txt          # Python version
├── .env                 # API keys (never committed)
└── knowledge_base/      # .txt documents Spark learns from
    ├── email_writing.txt
    ├── performance_reviews.txt
    ├── feedback_conversations.txt
    └── career_growth.txt
```

## API Routes

### Chat
| Method | Route | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/chat` | Send a message, get a response or draft trigger |
| POST | `/draft_answer` | Submit answer to draft question, get next question or document |
| POST | `/mirror` | Analyse writing samples, return communication fingerprint |

### Admin (Knowledge Base)
| Method | Route | Description |
|---|---|---|
| GET | `/admin/documents` | List all documents in knowledge base |
| POST | `/admin/paste` | Add a text entry to knowledge base |
| POST | `/admin/upload` | Upload a .txt or .pdf file |
| POST | `/admin/delete` | Delete a document by filename |
| POST | `/admin/rebuild` | Manually rebuild the vector store |

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOURUSERNAME/spark-backend.git
cd spark-backend
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Build the knowledge base
```bash
python rag_engine.py
```
This embeds all documents in `knowledge_base/` into ChromaDB. Only needed once, or when you add new documents manually.

### 6. Start the server
```bash
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000` — you should see:
```json
{"status": "Spark is running"}
```

## How the RAG Pipeline Works

```
User question
     ↓
Search ChromaDB for relevant chunks from knowledge_base/
     ↓
Send top 3 chunks + question to Llama 3 via Groq
     ↓
LLM generates answer grounded in your documents
     ↓
Return answer to frontend
```

## How the Document Drafter Works

```
User says "write a performance review"
     ↓
Intent detection identifies doc_type = performance_review
     ↓
Backend returns first question
     ↓
User answers each question (3 total)
     ↓
All answers sent to LLM with generation prompt
     ↓
Complete document returned, ready to copy
```

## Supported Document Types

- Performance Review
- Feedback Message
- Stakeholder Email
- Self Appraisal
- 1-on-1 Agenda

## Environment Variables

| Variable | Description | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | LLM API key | [console.groq.com](https://console.groq.com) |

## Deployment

Designed to deploy on [Render](https://render.com) free tier:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variable:** Add `GROQ_API_KEY` in Render dashboard

Note: Render free tier sleeps after 15 minutes of inactivity. First request after idle takes ~30 seconds to wake up.

## Adding to the Knowledge Base

Drop any `.txt` file into `knowledge_base/` and re-run:
```bash
python rag_engine.py
```

Or use the admin API routes to add content without restarting the server.

## License

MIT
