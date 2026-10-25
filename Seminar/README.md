# RAGnition — Intelligent Document Retrieval & Question Answering

> **Tagline:** Retrieve knowledge. Generate grounded answers.  
> **Architecture:** Modern AI SaaS Dashboard with Lexical DPR Baseline, CRAG Heuristic Correction, and Citation-Grounded Synthesis.

---

## 1. Project Overview

**RAGnition** is an educational, research-grade Retrieval-Augmented Generation (RAG) application engineered with a futuristic AI SaaS interface. 

Large Language Models (LLMs) suffer from hallucinations and static knowledge cutoff dates. RAGnition demonstrates how decomposing the problem into:
1. **Paragraph-Aware Document Chunking & Indexing**
2. **Lexical Evidence Retrieval** (Robertson-Spärck Jones IDF and BM25 term weighting)
3. **Corrective RAG (CRAG) Confidence Assessment & Expansion**
4. **Sentence Grounding & Source Attribution**

enables verifiable, hallucination-resistant answers with interactive citation anchors and zero external API dependencies.

---

## 2. Key Features & SaaS Redesign

- **Futuristic AI SaaS Dashboard:** Built on an intentional dark palette:
  - Deep Navy Background (`#0B1020`)
  - Elevated Dark Blue Cards (`#171F35`)
  - Secondary Surfaces (`#11182B`)
  - Electric Violet Accents (`#8B5CF6`)
  - Cyan Highlights & Glows (`#22D3EE`)
  - Slate Typography (`#F1F5F9`, `#94A3B8`)
- **Live Metric Statistics Bar:** Real-time metrics derived from active corpus state:
  - Active Document Count
  - Total Indexed Chunks
  - Corpus Vocabulary & Word Count
  - Live Query Latency in milliseconds
- **Preloaded Multi-Domain Reference Corpus:** Three complete, authoritative educational reference documents:
  1. *Retrieval-Augmented Generation (RAG)*
  2. *Artificial Intelligence and Machine Learning*
  3. *Climate Change: Causes, Mechanisms, and Impacts*
- **Question Console with Example Chips:** Focused input experience with keyboard shortcuts (`Enter` to submit) and interactive suggested query chips.
- **Visual Pipeline Execution Stepper:** Illuminated progress indicators tracking:
  `Query Parse` &rarr; `BM25 Retrieval` &rarr; `CRAG Check` &rarr; `Grounded Answer`
- **Expandable Evidence Cards:** Retrieved evidence cards display document names, chunk IDs, and derived BM25 scores. Each card supports inline Expand/Collapse toggling as well as a full context modal.
- **Copy Answer Functionality:** One-click clipboard copy with temporary feedback state.
- **CRAG-Inspired Retrieval Correction:** Evaluates retrieval confidence; automatically performs deterministic query broadening and comparative secondary retrieval passes if initial evidence is weak.
- **Evidence-Grounded Sentence Synthesis:** Extracts and ranks informative sentences with clickable citation pills `[DOC-ID-CHK#]`.
- **Hallucination Prevention (Refusal on Insufficient Evidence):** When out-of-domain queries are detected, RAGnition strictly declines to answer, providing a clear refusal explanation instead of fabricating content.
- **Corpus Management:** Add custom documents via text paste or `.txt` file drag-and-drop, delete custom documents, and persist data in `localStorage`.
- **System Architecture SVG:** Dark-themed inline SVG flowchart illustrating the complete pipeline.
- **Research Methodology Matrix:** Comparative breakdown of DPR, CRAG, RAPTOR, and ColPali.

---

## 3. Technologies Used

- **HTML5:** Semantic document layout, dark-theme SVG architecture flowchart, modal dialogs.
- **CSS3:** Custom futuristic design system with CSS custom properties, backdrop blur filters, glowing gradients, and responsive grid layouts.
- **Vanilla JavaScript (ES6+):** Zero-dependency algorithmic implementation of tokenization, stop-word removal, suffix stemming, Robertson-Spärck Jones IDF / BM25 scoring, CRAG expansion, and citation synthesis.
- **Browser LocalStorage API:** Client-side persistence for custom added documents.

> **Zero External Dependencies:** No Node.js server, Python backend, API keys, or paid cloud services required to run or evaluate the core system.

---

## 4. How to Run the Application

### Direct Browser Launch (Recommended)
Simply open `index.html` in any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, Safari):
- **Windows:** Double-click `index.html` or run `Start-Process index.html` in PowerShell.
- **macOS:** Run `open index.html` in Terminal.
- **Linux:** Run `xdg-open index.html`.

### Automated Test Suite
To verify all retrieval pipelines, chunking, CRAG checks, and document operations from the command line:
```bash
node test_pipeline.js
```

---

## 5. How the Retrieval Pipeline Works

```text
User Question
    │
    ▼
1. Query Processing (Lowercasing, Tokenization, Stop-word Filtering, Stemming)
    │
    ▼
2. Document Chunking & Inverted Index (TF-IDF & BM25 Robertson-Spärck Jones IDF)
    │
    ▼
3. Evidence Retrieval (Top-3 Chunks Ranked by Lexical Relevance)
    │
    ▼
4. Relevance Assessment
    ├─► High Score / Strong Coverage ─────► 6. Evidence-Grounded Answer
    │                                              ▲
    └─► Weak Score / Low Coverage                  │ (if improved)
            │                                      │
            ▼                                      │
        5. CRAG Correction (Deterministic Expansion & Re-ranking)
            ├─► Evidence Improved ─────────────────┘
            │
            └─► Evidence Still Insufficient ───► Decline to Answer (Refusal)
```

1. **Query Processing:** Normalizes input, strips non-alphanumeric punctuation, filters 100+ English stop words, and applies suffix stemming.
2. **Lexical Evidence Retrieval:** Computes Robertson-Spärck Jones IDF and BM25 term weighting ($k_1=1.5, b=0.75$) across all indexed chunks in the active corpus.
3. **Relevance Assessment & CRAG:** Evaluates retrieval confidence against score and query coverage thresholds. Low-confidence triggers deterministic synonym expansion and a comparative secondary retrieval pass.
4. **Answer Grounding & Attribution:** Candidate sentences within top-ranked chunks are scored and extracted into a synthesized answer accompanied by interactive citation anchors (`[DOC-ID-CHK#]`).
5. **Abstention on Insufficient Evidence:** If no chunk exceeds the confidence threshold even after CRAG correction, the system returns a structured refusal, preventing hallucination.

---

## 6. Implemented Methods vs. Future Work

| Research Method | Primary Purpose | Status in Prototype | Notes / Distinction |
|---|---|---|---|
| **DPR (Dense Passage Retrieval)** | Retrieve passages via dense semantic embeddings | **Simplified Lexical Baseline Implemented** | Implemented using BM25/IDF term overlap. Full neural DPR using dual BERT bi-encoders is future work. |
| **CRAG (Corrective RAG)** | Assess retrieval confidence and correct poor evidence | **Basic Corrective Retrieval Implemented** | Implemented via deterministic query expansion, secondary pass, and comparative re-ranking. Web search & neural evaluator are future work. |
| **RAPTOR (Recursive Summaries)** | Multi-level retrieval across tree hierarchies of summaries | **Future Work (Not Implemented)** | Documented and visually compared in Methodology; not implemented in this single-tier chunk prototype. |
| **ColPali (Visual RAG)** | Visual document page retrieval using Vision-Language Models | **Future Work (Not Implemented)** | Documented in Methodology; current prototype is text-based and does not process visual page layouts. |

---

## 7. Known Limitations

1. **Lexical vs. Dense Semantic Matching:** The retrieval engine relies on lexical overlap and suffix stemming rather than 768-dimensional neural vector embeddings.
2. **Extractive vs. Generative Language Model:** Answers are synthesized using sentence-level extraction and ranking from retrieved chunks rather than an autoregressive LLM, ensuring the system remains free, private, offline-capable, and fully verifiable.
3. **Single-Node In-Memory Storage:** The index is built in browser memory upon loading. Custom documents are stored in browser `localStorage`.
