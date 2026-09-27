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

client = genai.Client()

# Free-tier accessible Flash models (strictly excluding 'pro' models with limit: 0)
ACTIVE_FLASH_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# ============================================================================
# 1. Resilient Failover & Hard-Bounded Rate Limiter
# ============================================================================
def extract_safe_delay(err_text: str) -> float:
    """Extracts delay from retryInfo or fallback message, hard-clamped to [2, 60] seconds."""
    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s?", err_text)
    if not match:
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_text)
    if match:
        try:
            val = float(match.group(1))
            return max(2.0, min(val + 1.0, 60.0))
        except (ValueError, OverflowError):
            pass
    return 15.0


def safe_generate_content(contents: list, config: types.GenerateContentConfig):
    last_exception = None

    for model_name in ACTIVE_FLASH_MODELS:
        for attempt in range(3):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            except google.genai.errors.ClientError as e:
                last_exception = e
                err_msg = str(e)

                # Deprecated or not found
                if "404" in err_msg or "NOT_FOUND" in err_msg:
                    break

                # Model has zero free quota on this key -> switch to next model immediately
                if "limit: 0" in err_msg:
                    break

                # Genuine rate limit (429) -> apply bounded delay
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    wait_sec = extract_safe_delay(err_msg)
                    with st.spinner(f"⏳ Free quota cooling down ({model_name}). Resuming in {int(wait_sec)}s..."):
                        time.sleep(wait_sec)

            except google.genai.errors.ServerError as e:
                last_exception = e
                wait_sec = min(5.0 * (attempt + 1), 30.0)
                with st.spinner(f"⚡ {model_name} busy. Retrying in {int(wait_sec)}s..."):
                    time.sleep(wait_sec)

    raise last_exception


# ============================================================================
# 2. Pydantic Schemas
# ============================================================================
class BookKnowledgeBase(BaseModel):
    book_title_or_topic: str = Field(description="Inferred title, primary subject, or manuscript focus")
    chapter_structure_and_themes: List[str] = Field(description="Key sections, chapters, or foundational modules")
    visual_assets_catalog: List[str] = Field(description="Detected Kundalis, geometric charts, tables, or ephemerides with details")
    foundational_rules: List[str] = Field(description="Core universal principles, planetary significations, or primary formulas")
    exceptions_and_edge_cases: List[str] = Field(description="Conditional overrides, Neechbhanga, cancellations across chapters")
    full_ground_truth_summary: str = Field(description="Dense factual synthesis of the book's core teachings to anchor audits")

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
def save_kb_to_disk(filename: str, kb: BookKnowledgeBase):
    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename) + ".json"
    filepath = os.path.join(KB_STORAGE_DIR, clean_name)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(kb.model_dump_json(indent=2))
    return filepath

def load_kb_from_disk(filepath: str) -> BookKnowledgeBase:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BookKnowledgeBase.model_validate(data)

def get_saved_kb_files():
    if not os.path.exists(KB_STORAGE_DIR):
        return []
    return [f for f in os.listdir(KB_STORAGE_DIR) if f.endswith(".json")]


# ============================================================================
# 4. Ingestion & Agent Functions
# ============================================================================
def build_book_knowledge_base(book_file_ref: types.File) -> BookKnowledgeBase:
    system_prompt = """
    You are an expert Chief Archivist and Deep Document Synthesizer.
    Analyze this entire document/manuscript (including text, scanned diagrams, tables, and astrological charts).
    
    Compile a complete, high-fidelity Ground Truth Knowledge Base so downstream verification agents 
    can evaluate claims without reprocessing the visual document repeatedly.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=BookKnowledgeBase,
        temperature=0.1,
    )
    response = safe_generate_content(
        contents=[book_file_ref, "Index and synthesize the full structural knowledge base of this document."],
        config=config
    )
    return BookKnowledgeBase.model_validate_json(response.text)

def run_scholar(book_file_ref: Optional[types.File], kb: BookKnowledgeBase, user_query: str) -> ScholarReasoning:
    system_prompt = f"""
    You are Agent 1: 'The Scholar'. You analyze manuscripts and visual diagrams (astrological Kundalis, tables).
    
    ESTABLISHED KNOWLEDGE BASE:
    - Topic: {kb.book_title_or_topic}
    - Document Structure: {kb.chapter_structure_and_themes}
    - Visual Charts: {kb.visual_assets_catalog}
    - Key Exceptions Cataloged: {kb.exceptions_and_edge_cases}
    - Dense Ground Truth: {kb.full_ground_truth_summary}
    
    RULES:
    1. Base all deductions strictly on the document/knowledge base.
    2. Connect foundational rules with later-chapter overrides.
    3. State house positions, aspects, and planetary dignities clearly.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=ScholarReasoning,
        temperature=0.1,
    )
    contents = [f"Consult the knowledge base to answer: {user_query}"]
    if book_file_ref is not None:
        contents.insert(0, book_file_ref)
        
    response = safe_generate_content(contents=contents, config=config)
    return ScholarReasoning.model_validate_json(response.text)

def run_inquisitor(kb: BookKnowledgeBase, user_query: str, scholar_output: ScholarReasoning) -> InquisitorEvaluation:
    system_prompt = f"""
    You are Agent 2: 'The Inquisitor'. Your sole mission is to aggressively challenge and falsify Agent 1's claims.
    
    GROUND TRUTH KNOWLEDGE BASE:
    - Topic: {kb.book_title_or_topic}
    - Core Rules: {kb.foundational_rules}
    - Cataloged Exceptions & Edge Cases: {kb.exceptions_and_edge_cases}
    - Visual Asset Catalog: {kb.visual_assets_catalog}
    - Comprehensive Ground Truth: {kb.full_ground_truth_summary}
    
    ANTI-COLLUSION RULES:
    1. Challenge every assertion made by Agent 1 against this Knowledge Base.
    2. Check if Agent 1 applied a general rule while omitting an exception or edge case.
    3. Verify if chart placements match the cataloged visual assets.
    """
    eval_prompt = f"""
    USER INQUIRY: {user_query}
    
    AGENT 1 SUBMISSION:
    - Citations: {scholar_output.textual_citations}
    - Visual Observations: {scholar_output.visual_chart_observations}
    - Cross-Chapter Synthesis: {scholar_output.cross_chapter_synthesis}
    - Verdict: {scholar_output.final_verdict}
    
    Expose any discrepancy, over-generalization, or ungrounded assertion.
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
    
    AUTHORITATIVE GROUND TRUTH:
    - Topic: {kb.book_title_or_topic}
    - Foundational Teachings: {kb.foundational_rules}
    - Exceptions: {kb.exceptions_and_edge_cases}
    - Ground Truth Synthesis: {kb.full_ground_truth_summary}
    
    MANDATE:
    1. Detect collusion: Check whether Agent 2 gave an unearned pass to Agent 1.
    2. Verify against the Ground Truth to see which agent is right.
    3. Issue the definitive certified answer and a system integrity score (0-100).
    """
    audit_payload = f"""
    USER INQUIRY: {user_query}
    
    AGENT 1 REPORT:
    {scholar.model_dump_json(indent=2)}
    
    AGENT 2 EVALUATION:
    {inquisitor.model_dump_json(indent=2)}
    
    Audit both agents independently, verify chart readings against the source, check for collusion, and deliver the certified response.
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
# 5. Streamlit Frontend UI
# ============================================================================
st.title("🏛️ Deep Book Knowledge Base & Tri-Agent Deliberation Court")
st.markdown("Upload a book once to **generate and permanently save a Knowledge Base**, or select a previously saved index to query instantly.")

with st.sidebar:
    st.header("🗄️ Knowledge Base Mode")
    kb_mode = st.radio("Choose Source:", ["📂 Load Saved Knowledge Base", "📤 Index New Book/PDF"], index=0)

    if kb_mode == "📂 Load Saved Knowledge Base":
        saved_files = get_saved_kb_files()
        if saved_files:
            selected_file = st.selectbox("Select Cached Book Index:", saved_files)
            if st.button("⚡ Load Selected Index", use_container_width=True):
                loaded_kb = load_kb_from_disk(os.path.join(KB_STORAGE_DIR, selected_file))
                st.session_state["knowledge_base"] = loaded_kb
                st.session_state["active_file_name"] = selected_file.replace(".json", "")
                st.session_state["cloud_file"] = None
                st.success(f"Loaded: {selected_file}")
        else:
            st.info("No saved knowledge bases found yet. Choose 'Index New Book/PDF' to create your first one.")

    else:
        uploaded_file = st.file_uploader("Upload Scanned PDF / Book / Chart:", type=["pdf", "png", "jpg", "jpeg"])
        if uploaded_file is not None:
            if st.button("🚀 Process & Permanently Save Index", use_container_width=True):
                with st.spinner("Uploading document to AI Studio & compiling persistent Knowledge Base..."):
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
                    saved_path = save_kb_to_disk(uploaded_file.name, kb)

                    st.session_state["cloud_file"] = cloud_file
                    st.session_state["active_file_name"] = uploaded_file.name
                    st.session_state["knowledge_base"] = kb
                    st.success(f"Indexed and permanently saved to `{saved_path}`!")

    if "knowledge_base" in st.session_state:
        kb = st.session_state["knowledge_base"]
        st.divider()
        st.markdown("### 🗂️ Active Knowledge Base")
        st.write(f"**Focus:** {kb.book_title_or_topic}")
        with st.expander("Cataloged Visual Assets"):
            for v in kb.visual_assets_catalog:
                st.markdown(f"- `{v}`")
        with st.expander("Cross-Chapter Exceptions"):
            for e in kb.exceptions_and_edge_cases:
                st.markdown(f"- ⚠️ {e}")

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
        card1.warning("🧑‍🏫 **Agent 1: The Scholar**\n\n🔄 *Cross-referencing Knowledge Base & synthesizing rules...*")
        scholar_out = run_scholar(st.session_state.get("cloud_file"), st.session_state["knowledge_base"], user_query)
        card1.success("🧑‍🏫 **Agent 1: The Scholar**\n\n✅ *Synthesis Complete*")

        # Step 2: Inquisitor
        card2.warning("⚔️ **Agent 2: The Inquisitor**\n\n🔄 *Testing for missed exceptions & validating against Knowledge Base...*")
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