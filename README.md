# DefenseGPT — Defense Technology Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.x_LCEL-teal?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?style=flat&logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-LLaMA--3.3--70b_via_Groq-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> **DevelopersHub Corporation — AI/ML Engineering Internship | Task 4**

A context-aware Retrieval-Augmented Generation (RAG) chatbot specializing in defense technology, military aircraft, radar systems, UAVs, and Pakistani defense forces. Built with LangChain 1.x LCEL, FAISS vector search, and LLaMA-3.3-70b via Groq — deployed as an interactive Streamlit application.

---

## Overview

Standard LLMs hallucinate technical specifications and forget conversation context. **DefenseGPT** solves both problems:

1. **RAG (Retrieval-Augmented Generation)** — Every answer is grounded in a curated local knowledge base of 37 Wikipedia defense articles. The model cannot invent facts.
2. **Conversational Memory** — `RunnableWithMessageHistory` maintains session-based memory, enabling natural multi-turn dialogue where follow-up questions like *"What about its radar system?"* are understood in context.

---

## Features

- **Retrieval-Augmented Generation** using LangChain 1.x LCEL
- **History-aware retriever** — reformulates follow-up questions before searching FAISS
- **Local knowledge base** — 37 articles across 9 defense categories, built with a custom Wikipedia scraper
- **Session-based memory** with `RunnableWithMessageHistory`
- **Source citations** with direct Wikipedia links on every answer
- **Comparison queries** — structured side-by-side analysis (e.g. JF-17 vs F-16)
- **Dark / Light mode** toggle
- **Colored chat bubbles** — user (right, blue) and assistant (left, green)
- **Follow-up question suggestions** after each response
- **Enter key** to send messages
- Custom `DefenseGPT` system prompt with strict hallucination prevention

---

## System Architecture

```
OFFLINE — Knowledge Base Construction
──────────────────────────────────────────────────────────
Wikipedia Scraper (BeautifulSoup)
        ↓
Local .md files with YAML metadata (data/ folder)
        ↓
DirectoryLoader (LangChain)
        ↓
RecursiveCharacterTextSplitter (900 chars / 200 overlap)
        ↓
HuggingFace Embeddings (all-MiniLM-L6-v2, local)
        ↓
FAISS Vector Store (saved to faiss_defense_index/)

ONLINE — Inference Pipeline (LCEL)
──────────────────────────────────────────────────────────
User Query + Chat History
        ↓
History-Aware Retriever  ← reformulates follow-up questions
        ↓
FAISS Similarity Search (top-k chunks)
        ↓
Stuff Documents Chain    ← formats context + question
        ↓
Groq LLM (LLaMA-3.3-70b-versatile)
        ↓
Grounded Answer + Source Citations
        ↓
RunnableWithMessageHistory ← updates session memory
```

---

## Knowledge Base

| Category | Articles |
|---|---|
| 🇵🇰 Pakistan | Pakistan Army, PAF, Navy, ISI, ISPR, Operation Swift Retort, Kamra Aeronautical Complex |
| ✈️ Aircraft | JF-17 Thunder, F-16, Rafale, Eurofighter Typhoon, F-22 Raptor, Chengdu J-20 |
| 🚀 Missiles | Shaheen, Babur, Nasr, BrahMos, Hypersonic, Surface-to-Air, Missile Defense |
| 🛡️ Tanks | Al-Khalid, M1 Abrams, Leopard 2 |
| 📡 Radar | AESA Radar, AWACS, Phased Array |
| 🚁 Drones | UAV, MQ-9 Reaper |
| ⚙️ Technology | Stealth, Electronic Warfare, Cyber Warfare |
| 🚢 Naval | Aircraft Carrier, Submarine |
| 💡 Concepts | Air Superiority, Beyond-Visual-Range Missile, Network-Centric Warfare, Aerial Refueling |

**Total: 37 articles • ~1.5 million characters**

---

## Project Structure

```
defensegpt-rag-chatbot/
├── dataset_generator.ipynb        # Run once — builds local knowledge base
├── task4_defense_rag_lcel.ipynb   # Main RAG notebook (development)
├── app.py                         # Streamlit deployment
├── requirements.txt
├── .env                           # GROQ_API_KEY (do not commit)
├── .gitignore
├── README.md
├── data/                          # Local knowledge base (built by generator)
│   ├── pakistan/
│   ├── aircraft/
│   ├── missiles/
│   ├── tanks/
│   ├── radar/
│   ├── drones/
│   ├── technology/
│   ├── naval/
│   └── concepts/
└── faiss_defense_index/           # FAISS vector store (built by notebook)
    ├── index.faiss
    └── index.pkl
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/RabiyaMalik242/defensegpt-rag-chatbot.git
cd defensegpt-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Build the knowledge base (run once)
Open and run all cells in `dataset_generator.ipynb`. This scrapes Wikipedia and saves 37 articles as local `.md` files. Takes ~2 minutes.

### 5. Build the FAISS index
Open and run all cells in `task4_defense_rag_lcel.ipynb`. This embeds all documents and saves the FAISS index. Takes ~3–5 minutes on CPU.

### 6. Launch the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Example Conversations

**Direct factual query:**
```
User: What are the main specifications of the JF-17 Thunder?
DefenseGPT: The JF-17 Thunder is a lightweight multirole combat aircraft...
[Sources: ✈️ Aircraft — JF-17 Thunder]
```

**Follow-up with memory:**
```
User: Which countries operate it?
DefenseGPT: The JF-17 Thunder is currently operated by Pakistan and Myanmar...
```

**Comparison query:**
```
User: Compare the JF-17 and F-16 Fighting Falcon.
DefenseGPT: 
JF-17 Thunder:
- Manufacturer: PAC / Chengdu Aircraft Corporation
- Max Speed: Mach 1.6 ...

F-16 Fighting Falcon:
- Manufacturer: Lockheed Martin
- Max Speed: Mach 2.0 ...
```

**Out-of-corpus graceful refusal:**
```
User: What is the current oil price?
DefenseGPT: I couldn't find that in my knowledge base. Try asking about
military aircraft, radar systems, or defense technology instead.
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM Framework | LangChain 1.x LCEL |
| LLM | LLaMA-3.3-70b-versatile via Groq |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector Store | FAISS (local, saved to disk) |
| Memory | `RunnableWithMessageHistory` + `ChatMessageHistory` |
| Retriever | `create_history_aware_retriever` |
| QA Chain | `create_stuff_documents_chain` |
| Web Scraping | `requests` + `BeautifulSoup4` |
| Document Loading | `DirectoryLoader` + `TextLoader` |
| Frontend | Streamlit |
| Environment | `python-dotenv` |

---

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG) with LangChain 1.x LCEL
- Document embedding and FAISS vector search
- History-aware retrieval for multi-turn conversations
- Session-based conversational memory
- Custom Wikipedia scraper for local knowledge base construction
- LLM integration and prompt engineering
- Hallucination prevention via strict RAG grounding
- Production Streamlit deployment with dark/light mode

---

## Important Notes

- All answers are grounded in publicly available Wikipedia sources only
- This project is for educational purposes only
- Not intended for operational military use

---

## Author

**Rabiya Malik**
BS Software Engineering — Lahore Garrison University
AI/ML Engineering Intern @ DevelopersHub Corporation

[![GitHub](https://img.shields.io/badge/GitHub-RabiyaMalik242-black?style=flat&logo=github)](https://github.com/RabiyaMalik242)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/rabiya-malik)

---

## License

This project is licensed under the MIT License.
