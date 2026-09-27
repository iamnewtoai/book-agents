import json
import os
import re
import tempfile
import time
from typing import List, Optional
from google import genai
from google.genai import types
import google.genai.errors
import plotly.graph_objects as go
from pydantic import BaseModel, Field
import streamlit as st

st.set_page_config(
    page_title="Tri-Agent Deep Book Knowledge Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

KB_STORAGE_DIR = "knowledge_bases"
os.makedirs(KB_STORAGE_DIR, exist_ok=True)

# Initialize Google GenAI client (Reads GEMINI_API_KEY from environment)
client = genai.Client()

# Verified active model directly from your client.models.list()
MODEL_ID = "gemini-flash-latest"

# ============================================================================
# 1. Quota-Safe Request Wrapper
# ============================================================================
def safe_generate_content(contents: list, config: types.GenerateContentConfig):
    """Executes model inference with strictly clamped retry delay on quota rate limits."""
    for attempt in range(4):
        try:
            return client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config,
            )
        except google.genai.errors.ClientError as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                # Default wait is 15s, clamped between 5s and 45s
                wait_sec = 15.0
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                if match:
                    try:
                        wait_sec = min(max(float(match.group(1)) + 1.0, 5.0), 45.0)
                    except Exception:
                        wait_sec = 15.0
                
                with st.spinner(f"⏳ Free quota cooling down. Resuming in {int(wait_sec)}s..."):
                    time.sleep(wait_sec)
            else:
                raise e
        except google.genai.errors.ServerError as e:
            wait_sec = min(5.0 * (attempt + 1), 25.0)
            with st.spinner(f"⚡ Server busy. Retrying in {int(wait_sec)}s..."):
                time.sleep(wait_sec)

    return client.models.generate_content(model=MODEL_ID, contents=contents, config=config)


# ============================================================================
# 2. Schemas
# ============================================================================
class BookKnowledgeBase(BaseModel):
    book_title_or_topic: str = Field(description="Inferred title, primary subject, or manuscript focus")
    structural_outline: List[str] = Field(description="Summary of key chapters, sections, or divisions observed")
    visual_diagrams_found: List[str] = Field(description="Types of visual assets detected (e.g. Kundali charts, planetary degree tables)")
    foundational_principles: List[str] = Field(description="Core rules or primary definitions extracted")

class ScholarReasoning(BaseModel):
    textual_citations: List[str] = Field(description="Specific chapter, verse, or page citations from the book")
    visual_chart_observations: List[str] = Field(description="Exact planet, house, rashi, and aspect data read from the scanned chart/table")
    cross_chapter_synthesis: str = Field(description="How early foundation rules connect with later chapter rules")
    final_verdict: str = Field(description="Clear, reasoned answer to the user's inquiry")

class InquisitorEvaluation(BaseModel):
    unsubstantiated_claims: List[str] = Field(description="Statements made by Agent 1 not verified by the source text")
    contradictions_found: List[str] = Field(description="Exceptions or contradictory rules Agent 1 omitted")
    chart_reading_accuracy_pct: float = Field(description="Accuracy score of Agent 1's visual reading (0-100)")
    passed_inquisition: bool = Field(description="True only if reasoning has zero hallucinations")
    adversarial_critique: str = Field(description="Critical challenge exposing any flaws in Agent 1's work")

class AuditVerdict(BaseModel):
    collusion_detected: bool = Field(description="True if Agent 2 praised an inaccurate claim or overlooked obvious errors")
    system_integrity_score: float = Field(description="Final reliability score (0-100)")
    ground_truth_alignment: str = Field(description="Assessment of adherence to the original scanned book")
    certified_answer: str = Field(description="The final verified response delivered to the user")


# ============================================================================
# 3. Persistent Local Disk Storage
# ============================================================================
def save_kb_to_disk(filename: str, cloud_file_name: str, kb: BookKnowledgeBase):
    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename) + ".json"
    filepath = os.path.join(KB_STORAGE_DIR, clean_name)
    payload = {
        "cloud_file_name": cloud_file_name,
        "kb_data": kb.model_dump()
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return filepath

def load_kb_from_disk(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    kb = BookKnowledgeBase.model_validate(data["kb_data"])
    cloud_file_name = data.get("cloud_file_name")
    return kb, cloud_file_name

def get_saved_kb_files():
    if not os.path.exists(KB_STORAGE_DIR):
        return []
    return [f for f in os.listdir(KB_STORAGE_DIR) if f.endswith(".json")]


# ============================================================================
# 4. Ingestion & Agent Functions
# ============================================================================
def build_book_knowledge_base(book_file_ref: types.File) -> BookKnowledgeBase:
    """Efficient structural indexing that avoids blowing token limits."""
    system_prompt = """
    You are an expert Chief Archivist. Analyze the provided book or manuscript.
    Extract the title/subject, main structural sections, visual diagrams (Kundalis, tables), 
    and core foundational concepts into a clean Knowledge Base roadmap.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=BookKnowledgeBase,
        temperature=0.1,
    )
    response = safe_generate_content(
        contents=[book_file_ref, "Index the structural roadmap of this book."],
        config=config
    )
    return BookKnowledgeBase.model_validate_json(response.text)

def run_scholar(book_file_ref: Optional[types.File], kb: BookKnowledgeBase, user_query: str) -> ScholarReasoning:
    system_prompt = f"""
    You are Agent 1: 'The Scholar'. You analyze ancient texts, manuscripts, and astrological Kundali charts/tables.
    
    KNOWLEDGE BASE ROADMAP:
    - Topic: {kb.book_title_or_topic}
    - Outline: {kb.structural_outline}
    - Diagrams: {kb.visual_diagrams_found}
    - Core Principles: {kb.foundational_principles}
    
    RULES:
    1. Base all deductions strictly on the document.
    2. Synthesize concepts spanning across early and late chapters.
    3. If charts are present, state planetary positions, signs, houses, and aspects explicitly.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=ScholarReasoning,
        temperature=0.1,
    )
    contents = [f"Synthesize across the book to answer this question: {user_query}"]
    if book_file_ref is not None:
        contents.insert(0, book_file_ref)
        
    response = safe_generate_content(contents=contents, config=config)
    return ScholarReasoning.model_validate_json(response.text)

def run_inquisitor(kb: BookKnowledgeBase, user_query: str, scholar_output: ScholarReasoning) -> InquisitorEvaluation:
    system_prompt = f"""
    You are Agent 2: 'The Inquisitor'. Your sole mission is to aggressively challenge Agent 1's claims.
    
    KNOWLEDGE BASE ROADMAP:
    - Topic: {kb.book_title_or_topic}
    - Principles: {kb.foundational_principles}
    - Outline: {kb.structural_outline}
    
    ANTI-COLLUSION RULES:
    1. Do not agree with Agent 1 by default.
    2. Check for misidentified planetary dignities, forgotten cancellations (e.g. Neechbhanga), or missed exceptions.
    3. Expose any claim that over-generalizes without citing chapter context.
    """
    eval_prompt = f"""
    USER QUERY: {user_query}
    
    AGENT 1 REPORT:
    - Citations: {scholar_output.textual_citations}
    - Visual Observations: {scholar_output.visual_chart_observations}
    - Synthesis: {scholar_output.cross_chapter_synthesis}
    - Verdict: {scholar_output.final_verdict}
    
    Rigorously audit this work and expose any logical leaps, missed exceptions, or unverified claims.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=InquisitorEvaluation,
        temperature=0.2,
    )
    response = safe_generate_content(contents=[eval_prompt], config=config)
    return InquisitorEvaluation.model_validate_json(response.text)

def run_auditor(kb: BookKnowledgeBase, user_query: str, scholar: ScholarReasoning, inquisitor: InquisitorEvaluation) -> AuditVerdict:
    system_prompt = f"""
    You are Agent 3: 'The Supreme Auditor'. You report only to the user.
    
    ROADMAP GROUND TRUTH:
    - Topic: {kb.book_title_or_topic}
    - Principles: {kb.foundational_principles}
    
    MANDATE:
    1. Detect collusion: Did Agent 2 let flawed reasoning pass without rigorous testing?
    2. Settle disputes between Scholar and Inquisitor.
    3. Deliver the final certified answer and a system integrity score (0-100).
    """
    audit_payload = f"""
    USER QUERY: {user_query}
    
    AGENT 1 REPORT:
    {scholar.model_dump_json(indent=2)}
    
    AGENT 2 EVALUATION:
    {inquisitor.model_dump_json(indent=2)}
    
    Verify the debate, ensure zero collusion, and issue the final certified verdict.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=AuditVerdict,
        temperature=0.1,
    )
    response = safe_generate_content(contents=[audit_payload], config=config)
    return AuditVerdict.model_validate_json(response.text)

def create_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "System Integrity", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#00CC96" if score >= 75 else "#EF553B"},
            'steps': [
                {'range': [0, 50], 'color': "#2E1B1B"},
                {'range': [50, 75], 'color': "#332B1B"},
                {'range': [75, 100], 'color': "#1B3322"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=220, margin=dict(l=15, r=15, t=30, b=10))
    return fig


# ============================================================================
# 5. Streamlit User Interface
# ============================================================================
st.title("🏛️ Deep Book Knowledge Base & Tri-Agent Deliberation Court")
st.markdown("Upload a book once to **generate and permanently save a Knowledge Base**, or select a previously saved index to query instantly.")

with st.sidebar:
    st.header("⚙️ Active Backend")
    st.success(f"Connected Model: `{MODEL_ID}`")

    st.header("🗄️ Knowledge Base Mode")
    kb_mode = st.radio("Choose Source:", ["📂 Load Saved Knowledge Base", "📤 Index New Book/PDF"], index=0)

    if kb_mode == "📂 Load Saved Knowledge Base":
        saved_files = get_saved_kb_files()
        if saved_files:
            selected_file = st.selectbox("Select Cached Book Index:", saved_files)
            if st.button("⚡ Load Selected Index", use_container_width=True):
                loaded_kb, cloud_file_name = load_kb_from_disk(os.path.join(KB_STORAGE_DIR, selected_file))
                st.session_state["knowledge_base"] = loaded_kb
                st.session_state["active_file_name"] = selected_file.replace(".json", "")
                
                # Reconnect cloud reference if available
                if cloud_file_name:
                    try:
                        st.session_state["cloud_file"] = client.files.get(name=cloud_file_name)
                    except Exception:
                        st.session_state["cloud_file"] = None
                else:
                    st.session_state["cloud_file"] = None

                st.success(f"Loaded: {selected_file}")
        else:
            st.info("No saved knowledge bases found yet. Choose 'Index New Book/PDF' to create your first one.")

    else:
        uploaded_file = st.file_uploader("Upload Scanned PDF / Book / Chart:", type=["pdf", "png", "jpg", "jpeg"])
        if uploaded_file is not None:
            if st.button("🚀 Process & Permanently Save Index", use_container_width=True):
                with st.spinner("Uploading document to AI Studio & compiling Knowledge Base..."):
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    cloud_file = client.files.upload(file=tmp_path)
                    os.remove(tmp_path)

                    while cloud_file.state.name == "PROCESSING":
                        time.sleep(1)
                        cloud_file = client.files.get(name=cloud_file.name)

                    kb = build_book_knowledge_base(cloud_file)
                    saved_path = save_kb_to_disk(uploaded_file.name, cloud_file.name, kb)

                    st.session_state["cloud_file"] = cloud_file
                    st.session_state["active_file_name"] = uploaded_file.name
                    st.session_state["knowledge_base"] = kb
                    st.success(f"Indexed and permanently saved to `{saved_path}`!")

    if "knowledge_base" in st.session_state:
        kb = st.session_state["knowledge_base"]
        st.divider()
        st.markdown("### 🗂️ Active Knowledge Base")
        st.write(f"**Focus:** {kb.book_title_or_topic}")
        with st.expander("Detected Visual Diagrams"):
            for v in kb.visual_diagrams_found:
                st.markdown(f"- `{v}`")
        with st.expander("Foundational Principles"):
            for p in kb.foundational_principles:
                st.markdown(f"- 📌 {p}")

# Main Question & Answering Arena
if "knowledge_base" not in st.session_state:
    st.info("👈 Please load a saved Knowledge Base or index a new document from the sidebar to begin.")
else:
    st.write(f"📚 **Active Ground Truth:** `{st.session_state['active_file_name']}` — *Knowledge Base ready.*")

    user_query = st.text_area(
        "Enter your question (e.g. cross-chapter yoga synthesis, chart interpretations):",
        placeholder="Ex: Analyze the chart on page 15 along with Chapter 4 (Bhavas) and Chapter 12 (Yogas). What is the strength of the 10th lord and does any Neechbhanga yoga form?",
        height=90
    )

    ask_button = st.button("🚀 Consult Tri-Agent Court", use_container_width=True)

    if ask_button and user_query.strip():
        p1, p2, p3 = st.columns(3)
        card1 = p1.empty()
        card2 = p2.empty()
        card3 = p3.empty()

        card1.info("🧑‍🏫 **Agent 1: The Scholar**\n\n*Waiting in queue...*")
        card2.info("⚔️ **Agent 2: The Inquisitor**\n\n*Waiting in queue...*")
        card3.info("⚖️ **Agent 3: The Supreme Auditor**\n\n*Waiting in queue...*")

        # Step 1: Scholar
        card1.warning("🧑‍🏫 **Agent 1: The Scholar**\n\n🔄 *Cross-referencing rules & analyzing diagrams...*")
        scholar_out = run_scholar(st.session_state.get("cloud_file"), st.session_state["knowledge_base"], user_query)
        card1.success("🧑‍🏫 **Agent 1: The Scholar**\n\n✅ *Synthesis Complete*")

        # Step 2: Inquisitor
        card2.warning("⚔️ **Agent 2: The Inquisitor**\n\n🔄 *Testing for missed exceptions & challenging claims...*")
        inquisitor_out = run_inquisitor(st.session_state["knowledge_base"], user_query, scholar_out)
        if inquisitor_out.passed_inquisition:
            card2.success("⚔️ **Agent 2: The Inquisitor**\n\n✅ *Passed Rigorous Inquisition*")
        else:
            card2.error("⚔️ **Agent 2: The Inquisitor**\n\n⚠️ *Discrepancies / Omissions Flagged!*")

        # Step 3: Auditor
        card3.warning("⚖️ **Agent 3: The Supreme Auditor**\n\n🔄 *Checking for collusion & certifying ground truth...*")
        audit_out = run_auditor(st.session_state["knowledge_base"], user_query, scholar_out, inquisitor_out)
        card3.success("⚖️ **Agent 3: The Supreme Auditor**\n\n🏆 *Verdict Certified*")

        st.divider()

        # Top Row: Certified Verdict & Integrity Gauge
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("🏆 Certified Audited Response")
            st.success(audit_out.certified_answer)
            
            s1, s2 = st.columns(2)
            if audit_out.collusion_detected:
                s1.error("🚨 **Collusion Status:** Collusion / Rubber-Stamping Detected")
            else:
                s1.success("🛡️ **Collusion Status:** Verified Independent (Zero Collusion)")
            s2.info(f"**Ground Truth:** {audit_out.ground_truth_alignment}")

        with right_col:
            st.plotly_chart(create_gauge(audit_out.system_integrity_score), use_container_width=True)

        st.divider()

        # Detailed Breakdown Tabs
        t1, t2, t3 = st.tabs(["🧑‍🏫 Scholar Synthesis", "⚔️ Inquisitor Challenge", "🗂️ Full Audit Log"])

        with t1:
            st.markdown("#### 🔭 Visual Chart Observations")
            if scholar_out.visual_chart_observations:
                for obs in scholar_out.visual_chart_observations:
                    st.markdown(f"- `{obs}`")
            else:
                st.write("No visual diagrams required for this query.")
                
            st.markdown("#### 📖 Cross-Chapter Synthesis")
            st.write(scholar_out.cross_chapter_synthesis)
            
            st.markdown("#### 📑 Exact Citations")
            for c in scholar_out.textual_citations:
                st.markdown(f"- 📌 *{c}*")

        with t2:
            st.markdown("#### 🎯 Visual Chart Reading Fidelity")
            st.progress(int(inquisitor_out.chart_reading_accuracy_pct) / 100, text=f"Chart Reading Fidelity: {inquisitor_out.chart_reading_accuracy_pct}%")
            
            st.markdown("#### 🥊 Adversarial Critique")
            st.write(inquisitor_out.adversarial_critique)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Unsubstantiated Assertions:**")
                if inquisitor_out.unsubstantiated_claims:
                    for cl in inquisitor_out.unsubstantiated_claims:
                        st.error(f"❌ {cl}")
                else:
                    st.success("None detected.")
                    
            with col_b:
                st.markdown("**Contradictions / Overlooked Rules:**")
                if inquisitor_out.contradictions_found:
                    for co in inquisitor_out.contradictions_found:
                        st.warning(f"⚠️ {co}")
                else:
                    st.success("None detected.")

        with t3:
            st.json({
                "agent_1_scholar": scholar_out.model_dump(),
                "agent_2_inquisitor": inquisitor_out.model_dump(),
                "agent_3_auditor": audit_out.model_dump()
            })