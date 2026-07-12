"""
DefenseGPT — Defense Technology Knowledge Assistant
LangChain 1.x LCEL RAG Chatbot
Author: Rabiya Malik
"""

import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DefenseGPT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CATEGORY_META = {
    'pakistan':   {'icon': '🇵🇰', 'color': '#2ecc71', 'label': 'Pakistan'},
    'aircraft':   {'icon': '✈️',  'color': '#e74c3c', 'label': 'Aircraft'},
    'missiles':   {'icon': '🚀',  'color': '#e67e22', 'label': 'Missiles'},
    'tanks':      {'icon': '🛡️',  'color': '#95a5a6', 'label': 'Tanks'},
    'radar':      {'icon': '📡',  'color': '#3498db', 'label': 'Radar'},
    'drones':     {'icon': '🚁',  'color': '#f39c12', 'label': 'Drones'},
    'technology': {'icon': '⚙️',  'color': '#9b59b6', 'label': 'Technology'},
    'naval':      {'icon': '🚢',  'color': '#1abc9c', 'label': 'Naval'},
    'concepts':   {'icon': '💡',  'color': '#f1c40f', 'label': 'Concepts'},
}

DEFAULT_SUGGESTIONS = [
    "Compare the JF-17 Thunder and F-16 Fighting Falcon.",
    "What is AESA radar and how does it work?",
    "How does stealth technology reduce radar signature?",
    "What are the main roles of UAVs in modern warfare?",
    "Compare M1 Abrams and Leopard 2 tanks.",
    "What is Operation Swift Retort?",
]

FOLLOWUP_MAP = {
    'aircraft':   ["What radar does it use?", "Which countries operate it?", "How does it compare to the F-22?"],
    'pakistan':   ["What aircraft does the PAF operate?", "Tell me about Pakistan's missile program.", "What is ISPR?"],
    'missiles':   ["What is its range and payload?", "Which countries operate it?", "How does it compare to BrahMos?"],
    'tanks':      ["What is its armor type?", "How does it compare to the Leopard 2?", "What engine does it use?"],
    'radar':      ["Which aircraft use this radar?", "How does it detect stealth aircraft?", "What is its range?"],
    'drones':     ["What weapons can it carry?", "How are drones countered?", "What is its operational range?"],
    'technology': ["Which platforms use this?", "What are the countermeasures?", "How has this changed warfare?"],
    'naval':      ["How many aircraft can it carry?", "What is its propulsion system?", "Which navies operate it?"],
    'concepts':   ["Give me a real-world example.", "How does this apply to modern combat?", "Which countries use this doctrine?"],
}

# ─────────────────────────────────────────────
# MARKDOWN → HTML CONVERTER
# Converts LLM markdown responses for HTML bubble display
# ─────────────────────────────────────────────
def md_to_html(text: str) -> str:
    # Bold: **text** → <strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text*
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # Headers: ## / ###
    text = re.sub(r'^#{2,3}\s+(.+)$',
                  r'<br><span style="font-weight:700;font-size:1rem;">\1</span><br>',
                  text, flags=re.MULTILINE)
    # Bullet points
    text = re.sub(r'^[-•]\s+(.+)$',
                  r'<div style="padding:2px 0 2px 12px;">• \1</div>',
                  text, flags=re.MULTILINE)
    # Numbered lists
    text = re.sub(r'^\d+\.\s+(.+)$',
                  r'<div style="padding:2px 0 2px 12px;">\g<0></div>',
                  text, flags=re.MULTILINE)
    # Paragraph breaks
    text = re.sub(r'\n\n+', '<br><br>', text)
    # Single newlines
    text = text.replace('\n', '<br>')
    return text

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
def apply_css(dark: bool):
    if dark:
        bg, sidebar_bg  = '#0d1117', '#161b22'
        text, subtext   = '#e6edf3', '#8b949e'
        border          = '#30363d'
        input_bg        = '#21262d'
        btn_bg          = '#1f4e79'
        separator       = 'rgba(48,54,61,0.8)'
        user_bubble_bg  = 'linear-gradient(135deg, #1f4e79, #154360)'
        user_border     = '#3498db'
        asst_bubble_bg  = 'linear-gradient(135deg, #1b4332, #0d2818)'
        asst_border     = '#2ecc71'
        bubble_text     = '#e6edf3'
    else:
        bg, sidebar_bg  = '#f0f4f8', '#e8edf2'
        text, subtext   = '#1e293b', '#64748b'
        border          = '#cbd5e1'
        input_bg        = '#ffffff'
        btn_bg          = '#1d4ed8'
        separator       = 'rgba(203,213,225,0.8)'
        user_bubble_bg  = 'linear-gradient(135deg, #1d4ed8, #1e40af)'
        user_border     = '#3b82f6'
        asst_bubble_bg  = 'linear-gradient(135deg, #065f46, #064e3b)'
        asst_border     = '#10b981'
        bubble_text     = '#ffffff'

    st.session_state['_theme'] = {
        'bg': bg, 'sidebar_bg': sidebar_bg, 'text': text, 'subtext': subtext,
        'border': border, 'input_bg': input_bg, 'btn_bg': btn_bg,
        'separator': separator, 'user_bubble_bg': user_bubble_bg,
        'user_border': user_border, 'asst_bubble_bg': asst_bubble_bg,
        'asst_border': asst_border, 'bubble_text': bubble_text,
    }

    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg};
            color: {text};
            font-family: 'Segoe UI', system-ui, sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            border-right: 1px solid {border};
        }}
        [data-testid="stSidebar"] * {{ color: {text} !important; }}
        [data-testid="stHeader"] {{ display: none !important; }}
        .block-container {{ padding-top: 1rem !important; padding-bottom: 1rem !important; }}
        [data-testid="stAppViewContainer"] > .main {{ padding-top: 0.5rem !important; }}

        /* Input */
        .stTextInput input, [data-testid="stTextInput"] input {{
            background-color: {input_bg} !important;
            color: {text} !important;
            border: 1.5px solid {border} !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            font-size: 0.95rem !important;
        }}
        .stTextInput input:focus {{
            border-color: {user_border} !important;
            box-shadow: 0 0 0 2px {user_border}33 !important;
        }}

        /* Buttons */
        .stFormSubmitButton button, .stButton button {{
            background: linear-gradient(135deg, {btn_bg}, {btn_bg}cc) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            height: 46px !important;
        }}
        .stFormSubmitButton button:hover, .stButton button:hover {{
            opacity: 0.85 !important;
        }}

        /* Message separator */
        .msg-sep {{
            border: none;
            border-top: 1px solid {separator};
            margin: 14px 0 16px 0;
        }}

        /* Suggestion chips */
        .sugg-btn button {{
            background: {input_bg} !important;
            color: {subtext} !important;
            border: 1px solid {border} !important;
            border-radius: 20px !important;
            font-size: 0.82rem !important;
            height: auto !important;
            padding: 6px 14px !important;
            white-space: normal !important;
            text-align: left !important;
        }}
        .sugg-btn button:hover {{
            border-color: {user_border} !important;
            color: {user_border} !important;
        }}

        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: #2ecc71 !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background: {input_bg} !important;
            border-radius: 8px !important;
        }}

        /* Hide Streamlit branding */
        footer {{ display: none !important; }}
        #MainMenu {{ visibility: hidden !important; }}
        [data-testid="stToolbar"] {{ display: none !important; }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT BUBBLE RENDERERS
# ─────────────────────────────────────────────
def render_user_bubble(content: str):
    t = st.session_state.get('_theme', {})
    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: flex-end;
        margin: 10px 0 4px 0;
        padding: 0 6px;
    ">
        <div style="
            background: {t.get('user_bubble_bg', '#1f4e79')};
            border-left: 4px solid {t.get('user_border', '#3498db')};
            border-radius: 14px 14px 4px 14px;
            padding: 13px 18px;
            max-width: 75%;
            color: {t.get('bubble_text', '#e6edf3')};
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            font-size: 0.95rem;
            line-height: 1.55;
        ">
            <div style="font-size:0.76rem; opacity:0.65; margin-bottom:7px; font-weight:600; letter-spacing:0.3px;">
                👤 You
            </div>
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_assistant_bubble(content: str, category: str = None):
    t       = st.session_state.get('_theme', {})
    cat_m   = CATEGORY_META.get(category or '', {})
    badge   = ''
    if cat_m:
        badge = (
            f'<span style="display:inline-block;padding:1px 9px;border-radius:10px;'
            f'font-size:10px;font-weight:600;margin-bottom:8px;'
            f'background:{cat_m["color"]}25;color:{cat_m["color"]};'
            f'border:1px solid {cat_m["color"]}40;">'
            f'{cat_m["icon"]} {cat_m["label"]}</span>'
        )
    html_content = md_to_html(content)
    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: flex-start;
        margin: 10px 0 4px 0;
        padding: 0 6px;
    ">
        <div style="
            background: {t.get('asst_bubble_bg', '#1b4332')};
            border-left: 4px solid {t.get('asst_border', '#2ecc71')};
            border-radius: 14px 14px 14px 4px;
            padding: 13px 18px;
            max-width: 82%;
            color: {t.get('bubble_text', '#e6edf3')};
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            font-size: 0.95rem;
            line-height: 1.6;
        ">
            <div style="font-size:0.76rem; opacity:0.65; margin-bottom:7px; font-weight:600; letter-spacing:0.3px;">
                🛡️ DefenseGPT
            </div>
            {badge}
            <div>{html_content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources: list):
    if not sources:
        return
    t = st.session_state.get('_theme', {})
    sb = t.get('subtext', '#8b949e')
    ib = t.get('input_bg', '#21262d')
    bd = t.get('border', '#30363d')

    html = f"""
    <div style="margin: 6px 0 4px 20px;">
        <p style="font-size:12px; font-weight:600; color:{sb}; margin:0 0 6px 0;">📚 Sources</p>
    """
    for s in sources:
        m   = CATEGORY_META.get(s['category'], {'icon': '🔹', 'color': '#888', 'label': s['category'].title()})
        lnk = (
            f'<a href="{s["url"]}" target="_blank" '
            f'style="color:#58a6ff; text-decoration:none; font-weight:500;">'
            f'{s["title"]}</a>'
            if s.get('url') else
            f'<span style="color:{sb};">{s["title"]}</span>'
        )
        html += (
            f'<div style="background:{ib}; border:1px solid {bd}; border-radius:8px; '
            f'padding:7px 13px; margin:4px 0; font-size:13px; color:{sb};">'
            f'<span style="color:{m["color"]};">{m["icon"]} {m["label"]}</span>'
            f' — {lnk}</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_followups(suggestions: list, idx: int):
    if not suggestions:
        return
    t  = st.session_state.get('_theme', {})
    sb = t.get('subtext', '#8b949e')
    st.markdown(
        f'<p style="font-size:12px; color:{sb}; margin:10px 0 5px 20px;">'
        f'💬 Follow-up questions:</p>',
        unsafe_allow_html=True
    )
    cols = st.columns(len(suggestions[:3]))
    for k, fq in enumerate(suggestions[:3]):
        with cols[k]:
            st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
            if st.button(fq, key=f'fq_{idx}_{k}', use_container_width=True):
                st.session_state['pending_query'] = fq
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
def render_header(dark: bool):
    hdr_bg = ('linear-gradient(135deg,#080c14 0%,#0d2137 45%,#12092a 100%)'
               if dark else
               'linear-gradient(135deg,#1e3a5f 0%,#1a4f7a 45%,#2d1b69 100%)')
    tc   = '#e6edf3' if dark else '#ffffff'
    sc   = '#8b949e' if dark else 'rgba(255,255,255,0.75)'
    glow = 'rgba(52,152,219,0.35)'
    bdr  = 'rgba(52,152,219,0.25)' if dark else 'rgba(255,255,255,0.2)'
    bdgs = ''.join(
        f'<span style="display:inline-block;padding:3px 11px;border-radius:12px;'
        f'font-size:11px;font-weight:600;margin:3px;background:{m["color"]}20;'
        f'color:{m["color"]};border:1px solid {m["color"]}40;">'
        f'{m["icon"]} {m["label"]}</span>'
        for m in CATEGORY_META.values()
    )
    st.markdown(f"""
    <div style="background:{hdr_bg};border:1px solid {bdr};border-radius:14px;
                padding:28px 32px 22px 32px;margin-bottom:20px;position:relative;
                overflow:hidden;box-shadow:0 8px 32px {glow};">
        <div style="position:absolute;top:-40px;right:-40px;width:220px;height:220px;
                    background:radial-gradient(circle,{glow} 0%,transparent 70%);
                    pointer-events:none;"></div>
        <div style="position:absolute;bottom:-50px;left:35%;width:160px;height:160px;
                    background:radial-gradient(circle,rgba(46,204,113,0.07) 0%,transparent 70%);
                    pointer-events:none;"></div>
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:10px;">
            <div style="font-size:2.8rem;filter:drop-shadow(0 0 14px {glow});line-height:1;">🛡️</div>
            <div>
                <div style="font-size:1.72rem;font-weight:800;color:{tc};
                            letter-spacing:-0.5px;line-height:1.2;
                            text-shadow:0 2px 14px {glow};">
                    Defense Technology Knowledge Assistant
                </div>
                <div style="font-size:0.87rem;color:{sc};margin-top:6px;
                            display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <span>⚡ DefenseGPT</span><span style="opacity:0.35;">•</span>
                    <span>LangChain 1.x LCEL</span><span style="opacity:0.35;">•</span>
                    <span>37 defense articles</span><span style="opacity:0.35;">•</span>
                    <span>LLaMA-3.3-70b via Groq</span>
                </div>
            </div>
        </div>
        <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:2px;">{bdgs}</div>
        <div style="position:absolute;bottom:0;left:0;right:0;height:2px;
                    background:linear-gradient(90deg,transparent,#3498db 30%,
                    #2ecc71 60%,#9b59b6 85%,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RAG PROMPTS
# ─────────────────────────────────────────────
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ('system',
     'Given the chat history and the latest user question, '
     'reformulate the question to be fully self-contained without the chat history. '
     'Do NOT answer — only reformulate if needed, otherwise return as-is.'),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}'),
])

DEFENSE_SYSTEM_PROMPT = """
You are DefenseGPT, an expert AI assistant specializing in military aviation,
defense technology, weapons systems, radar, UAVs, naval systems, and Pakistani
defense forces.

You have access to a curated knowledge base covering fighter aircraft, missiles,
tanks, radar systems, drones, stealth technology, electronic warfare, and more.

STRICT RULES:
1. Answer ONLY using the retrieved context provided below.
2. NEVER invent specifications, ranges, speeds, or operational details.
3. If the answer is not in the context, respond with:
   "I couldn't find that in my knowledge base. Try asking about [suggest a related topic]."
4. If the user greets you (e.g. hi, hello, hey), respond warmly and briefly,
   then guide them: "Try asking about a specific defense topic such as military
   aircraft, radar systems, missiles, tanks, UAVs, or naval systems."
5. Answer professionally and precisely.
6. For specification questions: Present key specs in a clean structured format.
7. For comparison questions: Use a clear side-by-side structure covering
   the same attributes for each system (e.g. speed, range, engine, operators).
8. For historical/operational questions: Give context, then key facts.
9. Always mention which platform or system your answer refers to.
10. Cite the source document title where relevant.
11. Keep answers focused — detailed but not padded.

TONE: Professional, precise, technical. Like a defense analyst briefing.

Context:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ('system', DEFENSE_SYSTEM_PROMPT),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}'),
])

# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=False)
def load_vectorstore(_embeddings):
    if not os.path.exists('faiss_defense_index/index.faiss'):
        return None
    return FAISS.load_local(
        'faiss_defense_index', _embeddings,
        allow_dangerous_deserialization=True
    )

def build_chain(vectorstore, groq_api_key: str, top_k: int):
    llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.2, max_tokens=1024,
        groq_api_key=groq_api_key
    )
    retriever = vectorstore.as_retriever(
        search_type='similarity', search_kwargs={'k': top_k}
    )
    har = create_history_aware_retriever(llm, retriever, CONTEXTUALIZE_Q_PROMPT)
    qac = create_stuff_documents_chain(llm, QA_PROMPT)
    return create_retrieval_chain(har, qac)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_session():
    for k, v in {
        'messages': [], 'chat_store': {},
        'query_count': 0, 'dark_mode': True, '_theme': {}
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state['chat_store']:
        st.session_state['chat_store'][session_id] = ChatMessageHistory()
    return st.session_state['chat_store'][session_id]

SESSION_ID = 'defense_chat'

# ─────────────────────────────────────────────
# SOURCE HELPERS
# ─────────────────────────────────────────────
def extract_wiki_url(page_content: str) -> str:
    if not page_content.startswith('---'):
        return ''
    try:
        for line in page_content.split('\n')[1:]:
            if line.strip() == '---':
                break
            if line.startswith('source:'):
                url = line.replace('source:', '').strip()
                if url.startswith('http'):
                    return url
    except Exception:
        pass
    return ''

def extract_sources(context_docs):
    seen, sources = set(), []
    for doc in context_docs:
        title = doc.metadata.get('title', 'Unknown')
        if title not in seen:
            seen.add(title)
            sources.append({
                'title':    title,
                'category': doc.metadata.get('category', 'unknown').lower(),
                'url':      extract_wiki_url(doc.page_content),
            })
    return sources

def get_dominant_category(context_docs):
    counts = {}
    for doc in context_docs:
        c = doc.metadata.get('category', 'unknown').lower()
        counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get) if counts else None

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    init_session()
    dark = st.session_state['dark_mode']
    apply_css(dark)

    # ── Resolve defaults before sidebar renders ──
    groq_api_key = os.getenv('GROQ_API_KEY', '')
    top_k        = 4
    show_sources = True

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        if st.button('☀️ Light Mode' if dark else '🌙 Dark Mode',
                     use_container_width=True):
            st.session_state['dark_mode'] = not dark
            st.rerun()

        st.markdown('---')
        t  = st.session_state.get('_theme', {})
        sb = t.get('subtext', '#8b949e')

        st.markdown(
            f"<p style='font-size:12px;font-weight:600;color:{sb};"
            f"text-transform:uppercase;letter-spacing:1px;'>Knowledge Base</p>",
            unsafe_allow_html=True
        )
        for cat, meta in CATEGORY_META.items():
            st.markdown(
                f'<span style="color:{meta["color"]};">{meta["icon"]}</span> '
                f'<span style="font-size:13px;">{meta["label"]}</span>',
                unsafe_allow_html=True
            )

        st.markdown('---')
        st.markdown(
            f"<p style='font-size:12px;font-weight:600;color:{sb};"
            f"text-transform:uppercase;letter-spacing:1px;'>Session</p>",
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric('Queries', st.session_state.query_count)
        with c2:
            turns = len(get_session_history(SESSION_ID).messages) // 2
            st.metric('Memory', f'{turns} turns')

        st.markdown('---')
        with st.expander('⚙️ Settings'):
            key_inp = st.text_input(
                'Groq API Key', value=groq_api_key,
                type='password', help='Overrides .env key'
            )
            if key_inp:
                groq_api_key = key_inp
            top_k        = st.slider('Top-K Chunks', 2, 8, 4)
            show_sources = st.toggle('Show Source Citations', value=True)

        st.markdown('---')
        if st.button('🗑️ Clear Conversation', use_container_width=True):
            st.session_state.messages    = []
            st.session_state.chat_store  = {}
            st.session_state.query_count = 0
            st.rerun()

        st.markdown(
            f"<p style='font-size:11px;color:{'#555' if dark else '#94a3b8'};"
            f"margin-top:12px;text-align:center;'>"
            f"DevelopersHub AI/ML Internship<br>Task 4 — RAG Chatbot</p>",
            unsafe_allow_html=True
        )

    # ── Header ───────────────────────────────
    render_header(dark)

    # ── API key guard ─────────────────────────
    if not groq_api_key:
        st.warning('⚠️ No Groq API key found. Add it to `.env` or enter it in ⚙️ Settings.')
        st.info('Get a free key at https://console.groq.com')
        return

    # ── Load models ──────────────────────────
    with st.spinner('Loading knowledge base...'):
        embeddings  = load_embeddings()
        vectorstore = load_vectorstore(embeddings)

    if vectorstore is None:
        st.error('❌ FAISS index not found. Run the main notebook first to build it.')
        return

    rag_chain  = build_chain(vectorstore, groq_api_key, top_k)
    conv_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    # ── Suggested questions (empty state) ────
    if not st.session_state.messages:
        st.markdown(
            f"<p style='font-size:13px;color:{sb};margin-bottom:10px;'>"
            f"💡 <strong>Suggested Questions</strong></p>",
            unsafe_allow_html=True
        )
        cols = st.columns(3)
        for i, s in enumerate(DEFAULT_SUGGESTIONS):
            with cols[i % 3]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button(s, key=f's_{i}', use_container_width=True):
                    st.session_state['pending_query'] = s
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('---')

    # ── Chat history ─────────────────────────
    for idx, msg in enumerate(st.session_state.messages):
        if msg['role'] == 'user':
            render_user_bubble(msg['content'])
        else:
            render_assistant_bubble(msg['content'], msg.get('category'))
            if show_sources and msg.get('sources'):
                render_sources(msg['sources'])
            if msg.get('suggestions') and idx == len(st.session_state.messages) - 1:
                render_followups(msg['suggestions'], idx)
            st.markdown('<hr class="msg-sep">', unsafe_allow_html=True)

    # ── Input form (Enter key + button) ──────
    with st.form(key='chat_form', clear_on_submit=True):
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            user_input = st.text_input(
                'msg', label_visibility='collapsed',
                placeholder='Ask DefenseGPT about military technology...'
            )
        with col_btn:
            send = st.form_submit_button('Send ➤', use_container_width=True)

    if 'pending_query' in st.session_state:
        user_input = st.session_state.pop('pending_query')
        send = True

    # ── Process query ─────────────────────────
    if send and user_input and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({'role': 'user', 'content': query})

        with st.spinner('🔍 Searching knowledge base...'):
            try:
                resp    = conv_chain.invoke(
                    {'input': query},
                    config={'configurable': {'session_id': SESSION_ID}}
                )
                ctx     = resp.get('context', [])
                dom_cat = get_dominant_category(ctx)
                st.session_state.query_count += 1
                st.session_state.messages.append({
                    'role':        'assistant',
                    'content':     resp['answer'],
                    'sources':     extract_sources(ctx),
                    'category':    dom_cat,
                    'suggestions': FOLLOWUP_MAP.get(dom_cat, DEFAULT_SUGGESTIONS[:3]),
                })
            except Exception as e:
                st.error(f'Error: {e}')
                st.info('Check your Groq API key and ensure the FAISS index exists.')

        st.rerun()

    # ── Footer ────────────────────────────────
    st.markdown(
        f"<p style='text-align:center;font-size:11px;"
        f"color:{'#30363d' if dark else '#cbd5e1'};margin-top:20px;'>"
        f"DefenseGPT • Answers grounded in public Wikipedia sources • "
        f"Not for operational military use</p>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()