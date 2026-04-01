# DDR Report Generator

## Overview
An AI-powered pipeline that reads raw property inspection PDFs and generates a structured, client-ready **Detailed Diagnosis Report (DDR)** — complete with thermal images, area-wise observations, root causes, severity assessments, and recommended actions.

## Architecture

```
inputs/
  ├── Sample_Report.pdf         ← Site observations, checklists, issue descriptions
  └── Thermal_Images.pdf        ← IR thermography images + temperature readings

          │
          ▼
┌─────────────────────┐
│  PDF Ingestion       │  PyMuPDF extracts text + images
│  (src/ingestion/)    │  Images deduplicated by xref, saved to outputs/images/
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Text Chunking       │  LangChain RecursiveCharacterTextSplitter
│  + Metadata tagging  │  Each chunk tagged: source, page, doc_type, section
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Vector Store        │  ChromaDB (persistent, local)
│  (src/vectorstore/)  │  Embeddings: BAAI/bge-small-en-v1.5 (free, local)
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│  LangGraph Orchestration  (src/graph/)                │
│                                                       │
│  Node 1 → Property Issue Summary                     │
│  Node 2 → Area-wise Observations  (+images)           │
│  Node 3 → Probable Root Cause                        │
│  Node 4 → Severity Assessment                        │
│  Node 5 → Recommended Actions                        │
│  Node 6 → Additional Notes                           │
│  Node 7 → Missing / Unclear Information              │
│  Node 8 → Compile full report state                  │
└─────────┬────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  PDF Report Builder  │  ReportLab renders styled A4 PDF
│  (src/report/)       │  Embeds extracted images in correct sections
└─────────────────────┘
          │
          ▼
   outputs/DDR_Report_<timestamp>.pdf
```

## Tech Stack

| Component        | Technology                               |
|-----------------|------------------------------------------|
| Package manager  | `uv`                                     |
| LLM              | Cohere API (`command-r-plus-08-2024`)    |
| Embeddings       | `BAAI/bge-small-en-v1.5` (local, free)  |
| Vector DB        | ChromaDB (persistent, local)             |
| Orchestration    | LangGraph                                |
| PDF parsing      | PyMuPDF (`fitz`)                         |
| PDF generation   | ReportLab                                |
| Framework        | LangChain                                |

## Project Structure

```
ddr_report_generator/
├── pyproject.toml              # uv project config + dependencies
├── .env                        # Environment variables (API keys etc.)
├── .env.example                # Template for .env
├── .gitignore
├── README.md
├── main.py                     # CLI entry point
├── cleanup_images.py           # One-time cleanup of legacy duplicate images
│
├── src/
│   ├── config.py               # Centralised config from .env
│   ├── ingestion/
│   │   ├── pdf_parser.py       # Extract text + images (xref-deduped)
│   │   └── chunker.py          # Split text into overlapping chunks
│   ├── vectorstore/
│   │   ├── embedder.py         # Load BAAI/bge embedding model
│   │   └── store.py            # ChromaDB init, upsert, retrieval, clear
│   ├── graph/
│   │   ├── state.py            # LangGraph TypedDict state schema
│   │   ├── nodes.py            # One function per DDR section
│   │   ├── prompts.py          # All LLM prompt templates
│   │   └── pipeline.py         # Build + compile the LangGraph
│   ├── report/
│   │   ├── builder.py          # ReportLab PDF builder
│   │   └── styles.py           # Fonts, colours, layout constants
│   └── utils/
│       ├── logger.py           # Rich-based coloured logging
│       └── helpers.py          # Shared utility functions
│
├── inputs/                     # Drop your PDFs here
│   ├── Sample_Report.pdf
│   └── Thermal_Images.pdf
│
├── outputs/                    # Generated reports + extracted images
│   └── images/                 # *_xref<N>.png — deduplicated images
│
└── chroma_db/                  # Persistent ChromaDB embeddings
```

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) installed: `pip install uv`
- A [Cohere](https://dashboard.cohere.com/api-keys) API key (free tier works)

### 2. Setup

```bash
cd ddr_report_generator

# Install dependencies
uv sync

# Edit .env and set your Cohere key:
# COHERE_API_KEY=your_key_here
```

### 3. Add your PDFs

Place your two PDFs in the `inputs/` folder:
```
inputs/Sample_Report.pdf        ← inspection report
inputs/Thermal_Images.pdf       ← thermal / IR images document
```
(Or use different filenames and pass `--inspection` / `--thermal` flags.)

### 4. Run

```bash
# Standard run (skips re-ingestion if ChromaDB already has data)
uv run python main.py

# Custom PDF paths
uv run python main.py --inspection inputs/my_report.pdf --thermal inputs/thermal.pdf

# Force full re-ingestion (clears ChromaDB + re-processes + cleans legacy images)
uv run python main.py --reingest

# Clean legacy images without re-running the pipeline
uv run python cleanup_images.py
```

### 5. Output

```
outputs/DDR_Report_YYYYMMDD_HHMMSS.pdf
```

## DDR Output Sections

| # | Section | Contents |
|---|---------|----------|
| 1 | Property Issue Summary | High-level overview of all identified defects |
| 2 | Area-wise Observations | Issues by area (Bathroom, Balcony, Terrace, Wall…) with visual + thermal images |
| 3 | Probable Root Causes | AI-inferred root causes per area |
| 4 | Severity Assessment | Critical / High / Medium / Low with reasoning table |
| 5 | Recommended Actions | Specific repair treatments per area |
| 6 | Additional Notes | Precautions, limitations, follow-up suggestions |
| 7 | Missing / Unclear Info | Explicitly flags anything absent from source docs |

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `COHERE_API_KEY` | — | Your Cohere API key |
| `COHERE_MODEL` | `command-r-plus-08-2024` | Cohere model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local HuggingFace embedding model |
| `INSPECTION_PDF` | `Sample_Report.pdf` | Default inspection PDF filename |
| `THERMAL_PDF` | `Thermal_Images.pdf` | Default thermal PDF filename |
| `CHUNK_SIZE` | `800` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `150` | Chunk overlap (characters) |

## How Images Are Handled

1. **Extraction**: PyMuPDF walks every page and collects image xrefs. Each unique xref is saved **once** as `<doc_type>_xref<NNNNN>.png` — no per-page duplication.
2. **Section hints**: Each image is tagged with the area it belongs to (e.g. `bathroom`, `terrace`, `external_wall`) using keyword scoring from the page's text, with a page-range fallback for the thermal PDF.
3. **Assignment**: During report generation, each DDR area selects matching images using section_hint, with fallback to related areas if no direct match is found.
4. **Embedding**: Images appear as side-by-side pairs (visual evidence + thermal analysis) directly beneath their area's observations.

## Troubleshooting

**Too many images / old duplicate files**
```bash
uv run python cleanup_images.py
```

**ChromaDB out of sync**
```bash
uv run python main.py --reingest
```

**Cohere API errors**
- Check your key at https://dashboard.cohere.com/api-keys
- Make sure `COHERE_API_KEY` is set in `.env`

## Running Tests

```bash
uv run pytest tests/ -v
```
