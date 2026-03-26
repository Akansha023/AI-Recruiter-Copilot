import os
"""
AI Recruiter Copilot — Main Streamlit Application
Built with Gemini + RAG (ChromaDB + SentenceTransformers)
"""

import streamlit as st
import time
from utils import process_resume, retrieve_relevant_context, call_llm
import prompts

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Recruiter Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .badge {
        display: inline-block;
        background: #4f46e5;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px;
    }

    .result-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .score-display {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4f46e5;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
    }

    .sidebar-info {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #4a5568;
    }

    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Enter your API key...",
        help="Get your free key at https://aistudio.google.com"
    )
    if api_key:
        import os
        os.environ["GOOGLE_API_KEY"] = api_key
        # Reinitialize genai with new key
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        from utils import model
        st.success("✅ API Key set!")

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Enter your Gemini API key above
    2. Upload a candidate's resume (PDF)
    3. Paste the job description
    4. Click **Analyze Candidate**
    5. Review AI-generated insights!
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    for tech in ["Gemini 2.5 Flash (LLM)", "ChromaDB (Vector DB)", "SentenceTransformers", "LangChain (Chunking)", "Streamlit (UI)"]:
        st.markdown(f"• {tech}")

    st.markdown("---")
    st.markdown("""
    <div class='sidebar-info'>
    💡 <b>Free Stack:</b> All tools used are 100% free for development. Gemini API has a generous free tier.
    </div>
    """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Recruiter Copilot</h1>
    <p>Intelligent resume analysis powered by Gemini AI + RAG pipeline</p>
    <div>
        <span class="badge">Gemini AI</span>
        <span class="badge">RAG Pipeline</span>
        <span class="badge">Vector Search</span>
        <span class="badge">Free Stack</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Mode Selector ────────────────────────────────────────────────────────────
mode = st.radio(
    "Select Mode",
    ["🔍 Single Candidate Analysis", "⚖️ Compare Two Candidates"],
    horizontal=True
)

st.markdown("---")


# ─── SINGLE CANDIDATE MODE ────────────────────────────────────────────────────
if mode == "🔍 Single Candidate Analysis":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload the candidate's resume in PDF format"
        )
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")

    with col2:
        st.subheader("📋 Job Description")
        jd = st.text_area(
            "Paste the full job description here",
            height=200,
            placeholder="We are looking for a Senior Software Engineer with 5+ years of experience in Python, distributed systems, and cloud infrastructure..."
        )

    st.markdown("---")

    analyze_btn = st.button("🚀 Analyze Candidate", use_container_width=True)

    if analyze_btn:
        if not uploaded_file:
            st.error("⚠️ Please upload a resume PDF.")
        elif not jd.strip():
            st.error("⚠️ Please paste a job description.")
        elif not os.environ.get("GOOGLE_API_KEY", ""):
            st.error("⚠️ Please enter your Gemini API key in the sidebar.")
        else:
            # ─── Processing Pipeline ───────────────────────────────────────
            with st.spinner("🔄 Processing resume..."):
                progress = st.progress(0)
                status = st.empty()

                try:
                    status.text("📤 Extracting text from PDF...")
                    progress.progress(15)
                    raw_text, collection_name = process_resume(uploaded_file)
                    time.sleep(0.3)

                    status.text("🧠 Building semantic index...")
                    progress.progress(35)
                    context = retrieve_relevant_context(jd, collection_name)
                    time.sleep(0.3)

                    status.text("✍️ Generating summary...")
                    progress.progress(50)
                    summary = call_llm(prompts.summary_prompt(context))

                    status.text("🔍 Analyzing strengths & risks...")
                    progress.progress(65)
                    sr = call_llm(prompts.strengths_risks_prompt(context, jd))

                    status.text("🎯 Calculating fit score...")
                    progress.progress(80)
                    score = call_llm(prompts.scoring_prompt(context, jd))

                    status.text("💬 Generating interview questions...")
                    progress.progress(92)
                    questions = call_llm(prompts.questions_prompt(context, jd))

                    progress.progress(100)
                    status.text("✅ Analysis complete!")
                    time.sleep(0.5)
                    progress.empty()
                    status.empty()

                    # ─── Results ──────────────────────────────────────────
                    st.success("🎉 Analysis complete!")
                    st.markdown("---")

                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📌 Summary", "✅ Strengths & Risks",
                        "🎯 Fit Score", "💬 Interview Questions"
                    ])

                    with tab1:
                        st.markdown("### Candidate Summary")
                        st.markdown(summary)
                        with st.expander("📄 View Raw Resume Text"):
                            st.text(raw_text[:3000] + "..." if len(raw_text) > 3000 else raw_text)

                    with tab2:
                        st.markdown("### Strengths & Risks Analysis")
                        st.markdown(sr)

                    with tab3:
                        st.markdown("### Candidate Fit Score")
                        st.markdown(score)

                    with tab4:
                        st.markdown("### Interview Question Bank")
                        st.markdown(questions)

                except ValueError as ve:
                    st.error(f"❌ {str(ve)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.info("💡 Make sure your API key is valid and the PDF is not password-protected.")


# ─── COMPARE MODE ─────────────────────────────────────────────────────────────
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Candidate A")
        file_a = st.file_uploader("Upload Resume A (PDF)", type=['pdf'], key="a")
        if file_a:
            st.success(f"✅ {file_a.name}")

    with col2:
        st.subheader("👤 Candidate B")
        file_b = st.file_uploader("Upload Resume B (PDF)", type=['pdf'], key="b")
        if file_b:
            st.success(f"✅ {file_b.name}")

    jd_compare = st.text_area(
        "📋 Job Description (shared)",
        height=150,
        placeholder="Paste the job description here..."
    )

    compare_btn = st.button("⚖️ Compare Candidates", use_container_width=True)

    if compare_btn:
        if not file_a or not file_b:
            st.error("⚠️ Please upload both resumes.")
        elif not jd_compare.strip():
            st.error("⚠️ Please paste a job description.")
        elif not os.environ.get("GOOGLE_API_KEY", ""):
            st.error("⚠️ Please enter your Gemini API key in the sidebar.")
        else:
            with st.spinner("🔄 Comparing candidates..."):
                try:
                    _, col_a = process_resume(file_a)
                    _, col_b = process_resume(file_b)
                    ctx_a = retrieve_relevant_context(jd_compare, col_a)
                    ctx_b = retrieve_relevant_context(jd_compare, col_b)
                    comparison = call_llm(prompts.compare_prompt(ctx_a, ctx_b, jd_compare))

                    st.success("✅ Comparison complete!")
                    st.markdown("---")
                    st.markdown("### ⚖️ Head-to-Head Comparison")
                    st.markdown(comparison)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Candidate A — Fit Score")
                        score_a = call_llm(prompts.scoring_prompt(ctx_a, jd_compare))
                        st.markdown(score_a)
                    with col2:
                        st.markdown("#### Candidate B — Fit Score")
                        score_b = call_llm(prompts.scoring_prompt(ctx_b, jd_compare))
                        st.markdown(score_b)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; font-size: 0.8rem; padding: 1rem;'>
    Built with ❤️ using Gemini AI + RAG Pipeline &nbsp;|&nbsp; 
    <a href='https://github.com' style='color: #4f46e5;'>GitHub</a> &nbsp;|&nbsp;
    Stack: Streamlit · ChromaDB · SentenceTransformers · LangChain
</div>
""", unsafe_allow_html=True)
