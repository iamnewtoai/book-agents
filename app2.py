import json
import os
import tempfile
import time
from typing import List, Optional
from google import genai
from google.genai import types
import plotly.graph_objects as go
from pydantic import BaseModel, Field
import streamlit as st

# 1. Page Configuration (Corrected parameter name: layout)
st.set_page_config(
    page_title="Tri-Agent Deep Book Knowledge Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Google GenAI client (Reads GEMINI_API_KEY from environment)
client = genai.Client()
FREE_MODEL = "gemini-2.5-flash"

# ============================================================================
# 1. Schemas: Knowledge Base & Tri-Agent Deliberation
# ============================================================================


class BookKnowledgeBase(BaseModel):
  book_title_or_topic: str = Field(
      description="Inferred title, primary subject, or manuscript focus"
  )
  chapter_structure_and_themes: List[str] = Field(
      description="Key sections, chapters, or foundational modules discovered"
  )
  visual_assets_catalog: List[str] = Field(
      description=(
          "Detected Kundalis, geometric charts, tables, or ephemerides with"
          " their page/section references"
      )
  )
  foundational_rules: List[str] = Field(
      description=(
          "Core universal principles, planetary significations, or primary"
          " formulas"
      )
  )
  exceptions_and_edge_cases: List[str] = Field(
      description=(
          "Conditional overrides, Neechbhanga, cancellations, or contrasting"
          " rules found in later chapters"
      )
  )


class ScholarReasoning(BaseModel):
  textual_citations: List[str] = Field(
      description="Specific chapter, verse, or page citations from the book"
  )
  visual_chart_observations: List[str] = Field(
      description=(
          "Exact planet, house, rashi, and aspect data read from the scanned"
          " chart/table"
      )
  )
  cross_chapter_synthesis: str = Field(
      description="How early foundation rules connect with later chapter rules"
  )
  final_verdict: str = Field(
      description="Clear, reasoned answer to the user's inquiry"
  )


class InquisitorEvaluation(BaseModel):
  unsubstantiated_claims: List[str] = Field(
      description="Statements made by Agent 1 not verified by the source text"
  )
  contradictions_found: List[str] = Field(
      description="Exceptions or contradictory rules Agent 1 omitted"
  )
  chart_reading_accuracy_pct: float = Field(
      description="Accuracy score of Agent 1's visual reading (0-100)"
  )
  passed_inquisition: bool = Field(
      description="True only if reasoning has zero hallucinations"
  )
  adversarial_critique: str = Field(
      description="Critical challenge exposing any flaws in Agent 1's work"
  )


class AuditVerdict(BaseModel):
  collusion_detected: bool = Field(
      description=(
          "True if Agent 2 praised an inaccurate claim or overlooked obvious"
          " errors"
      )
  )
  system_integrity_score: float = Field(
      description="Final reliability score (0-100)"
  )
  ground_truth_alignment: str = Field(
      description="Assessment of adherence to the original scanned book"
  )
  certified_answer: str = Field(
      description="The final verified response delivered to the user"
  )


# ============================================================================
# 2. Knowledge Base Ingestion Pipeline (Run Once Per Document)
# ============================================================================
def build_book_knowledge_base(book_file_ref: types.File) -> BookKnowledgeBase:
  system_prompt = """
    You are an expert Chief Archivist and Deep Document Synthesizer.
    Analyze this entire document/manuscript (including text, scanned diagrams, tables, and astrological charts).
    
    TASKS:
    1. Extract chapter progression and thematic divisions.
    2. Catalog all visual charts (e.g. Kundali houses, planetary positions) and tables.
    3. Extract fundamental early-chapter definitions and later-chapter exceptions/overrides.
    Compile this into a structured Knowledge Base to anchor all future reasoning.
    """
  response = client.models.generate_content(
      model=FREE_MODEL,
      contents=[
          book_file_ref,
          "Index and synthesize the full structural knowledge base of this"
          " document.",
      ],
      config=types.GenerateContentConfig(
          system_instruction=system_prompt,
          response_mime_type="application/json",
          response_schema=BookKnowledgeBase,
          temperature=0.1,
      ),
  )
  return BookKnowledgeBase.model_validate_json(response.text)


# ============================================================================
# 3. Tri-Agent Execution Functions
# ============================================================================
def run_scholar(
    book_file_ref: types.File, kb: BookKnowledgeBase, user_query: str
) -> ScholarReasoning:
  system_prompt = f"""
    You are Agent 1: 'The Scholar'. You have profound expertise in cross-chapter text analysis, 
    ancient manuscripts, and visual diagrams (astrological Kundalis, tables, ephemerides).
    
    ESTABLISHED KNOWLEDGE BASE SUMMARY:
    - Topic: {kb.book_title_or_topic}
    - Document Structure: {kb.chapter_structure_and_themes}
    - Visual Charts: {kb.visual_assets_catalog}
    - Key Exceptions Cataloged: {kb.exceptions_and_edge_cases}
    
    RULES:
    1. Base all deductions strictly on the document.
    2. Connect early foundational rules with later-chapter overrides.
    3. For charts, explicitly state house positions, aspects, and planetary dignities.
    """
  response = client.models.generate_content(
      model=FREE_MODEL,
      contents=[
          book_file_ref,
          f"Consult the knowledge base and document to answer: {user_query}",
      ],
      config=types.GenerateContentConfig(
          system_instruction=system_prompt,
          response_mime_type="application/json",
          response_schema=ScholarReasoning,
          temperature=0.1,
      ),
  )
  return ScholarReasoning.model_validate_json(response.text)


def run_inquisitor(
    book_file_ref: types.File,
    kb: BookKnowledgeBase,
    user_query: str,
    scholar_output: ScholarReasoning,
) -> InquisitorEvaluation:
  system_prompt = f"""
    You are Agent 2: 'The Inquisitor'. Your sole mission is to aggressively challenge and falsify Agent 1's claims.
    
    KNOWLEDGE BASE REFERENCE:
    - Exceptions / Overrides: {kb.exceptions_and_edge_cases}
    - Visual Inventory: {kb.visual_assets_catalog}
    
    ANTI-COLLUSION RULES:
    1. Hunt for misread planetary symbols, house numbers, or degree tables.
    2. Check if Agent 1 cited a general rule while ignoring a specific exception in another chapter.
    3. Score the visual reading strictly against the actual chart images.
    """
  eval_prompt = f"""
    USER INQUIRY: {user_query}
    
    AGENT 1 SUBMISSION:
    - Citations: {scholar_output.textual_citations}
    - Chart Observations: {scholar_output.visual_chart_observations}
    - Synthesis: {scholar_output.cross_chapter_synthesis}
    - Verdict: {scholar_output.final_verdict}
    
    Inspect the attached source directly and expose every flaw or missed nuance.
    """
  response = client.models.generate_content(
      model=FREE_MODEL,
      contents=[book_file_ref, eval_prompt],
      config=types.GenerateContentConfig(
          system_instruction=system_prompt,
          response_mime_type="application/json",
          response_schema=InquisitorEvaluation,
          temperature=0.2,
      ),
  )
  return InquisitorEvaluation.model_validate_json(response.text)


def run_auditor(
    book_file_ref: types.File,
    user_query: str,
    scholar: ScholarReasoning,
    inquisitor: InquisitorEvaluation,
) -> AuditVerdict:
  system_prompt = """
    You are Agent 3: 'The Supreme Auditor'. You report only to the user.
    
    MANDATE:
    1. Detect collusion: Check whether Agent 2 gave an unearned pass to Agent 1.
    2. Check the raw scanned book independently to determine ground truth.
    3. Correct any inaccuracies and issue the certified final answer with an integrity score (0-100).
    """
  audit_payload = f"""
    USER INQUIRY: {user_query}
    
    AGENT 1 REPORT:
    {scholar.model_dump_json(indent=2)}
    
    AGENT 2 EVALUATION:
    {inquisitor.model_dump_json(indent=2)}
    
    Audit both agents independently, verify chart readings against the source, check for collusion, and deliver the certified response.
    """
  response = client.models.generate_content(
      model=FREE_MODEL,
      contents=[book_file_ref, audit_payload],
      config=types.GenerateContentConfig(
          system_instruction=system_prompt,
          response_mime_type="application/json",
          response_schema=AuditVerdict,
          temperature=0.1,
      ),
  )
  return AuditVerdict.model_validate_json(response.text)


# ============================================================================
# 4. Visualization Helpers
# ============================================================================
def create_gauge(score: float):
  fig = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=score,
          domain={"x": [0, 1], "y": [0, 1]},
          title={"text": "System Integrity", "font": {"size": 18}},
          gauge={
              "axis": {"range": [0, 100], "tickwidth": 1},
              "bar": {"color": "#00CC96" if score >= 75 else "#EF553B"},
              "steps": [
                  {"range": [0, 50], "color": "#2E1B1B"},
                  {"range": [50, 75], "color": "#332B1B"},
                  {"range": [75, 100], "color": "#1B3322"},
              ],
              "threshold": {
                  "line": {"color": "white", "width": 3},
                  "thickness": 0.75,
                  "value": score,
              },
          },
      )
  )
  fig.update_layout(height=220, margin=dict(l=15, r=15, t=30, b=10))
  return fig


# ============================================================================
# 5. Streamlit Frontend UI
# ============================================================================
st.title("🏛️ Deep Book Knowledge Base & Tri-Agent Deliberation Court")
st.markdown(
    "Upload a scanned manuscript, book, or Kundali once to **build an indexed"
    " Knowledge Base**. Then ask questions requiring cross-chapter reasoning,"
    " adversarial checks, and anti-collusion audits."
)

with st.sidebar:
  st.header("📖 Document Repository")
  uploaded_file = st.file_uploader(
      "Upload Scanned PDF / Book / Chart:", type=["pdf", "png", "jpg", "jpeg"]
  )

  # Check if a new file is uploaded
  if uploaded_file is not None:
    if (
        "active_file_name" not in st.session_state
        or st.session_state["active_file_name"] != uploaded_file.name
    ):
      with st.spinner(
          "Uploading document to AI Studio & compiling one-time Knowledge"
          " Base..."
      ):
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
          tmp.write(uploaded_file.read())
          tmp_path = tmp.name

        # Upload to Gemini File API
        cloud_file = client.files.upload(file=tmp_path)
        os.remove(tmp_path)

        while cloud_file.state.name == "PROCESSING":
          time.sleep(1)
          cloud_file = client.files.get(name=cloud_file.name)

        # Build Knowledge Base Once
        kb = build_book_knowledge_base(cloud_file)

        # Store in session state
        st.session_state["cloud_file"] = cloud_file
        st.session_state["active_file_name"] = uploaded_file.name
        st.session_state["knowledge_base"] = kb
        st.success(f"Indexed: {uploaded_file.name}")

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
  st.info("👈 Upload your scanned book or document in the sidebar to begin.")
else:
  st.write(
      f"📚 **Active Ground Truth:** `{st.session_state['active_file_name']}`"
      f" — *Knowledge Base ready in memory.*"
  )

  user_query = st.text_area(
      "Enter your question (e.g. cross-chapter yoga synthesis, chart interpretations):",
      placeholder=(
          "Ex: Analyze the chart on page 15 along with Chapter 4 (Bhavas) and"
          " Chapter 12 (Yogas). What is the strength of the 10th lord and does"
          " any Neechbhanga yoga form?"
      ),
      height=90,
  )

  ask_button = st.button("🚀 Consult Tri-Agent Court", use_container_width=True)

  if ask_button and user_query.strip():
    # Visual Deliberation Arena
    p1, p2, p3 = st.columns(3)
    card1 = p1.empty()
    card2 = p2.empty()
    card3 = p3.empty()

    card1.info("🧑‍🏫 **Agent 1: The Scholar**\n\n*Waiting in queue...*")
    card2.info("⚔️ **Agent 2: The Inquisitor**\n\n*Waiting in queue...*")
    card3.info("⚖️ **Agent 3: The Supreme Auditor**\n\n*Waiting in queue...*")

    # Step 1: Scholar
    card1.warning(
        "🧑‍🏫 **Agent 1: The Scholar**\n\n🔄 *Cross-referencing Knowledge Base &"
        " synthesizing rules...*"
    )
    scholar_out = run_scholar(
        st.session_state["cloud_file"],
        st.session_state["knowledge_base"],
        user_query,
    )
    card1.success("🧑‍🏫 **Agent 1: The Scholar**\n\n✅ *Synthesis Complete*")

    time.sleep(1)

    # Step 2: Inquisitor
    card2.warning(
        "⚔️ **Agent 2: The Inquisitor**\n\n🔄 *Testing for missed exceptions &"
        " validating visual charts...*"
    )
    inquisitor_out = run_inquisitor(
        st.session_state["cloud_file"],
        st.session_state["knowledge_base"],
        user_query,
        scholar_out,
    )
    if inquisitor_out.passed_inquisition:
      card2.success(
          "⚔️ **Agent 2: The Inquisitor**\n\n✅ *Passed Rigorous Inquisition*"
      )
    else:
      card2.error(
          "⚔️ **Agent 2: The Inquisitor**\n\n⚠️ *Discrepancies / Omissions"
          " Flagged!*"
      )

    time.sleep(1)

    # Step 3: Auditor
    card3.warning(
        "⚖️ **Agent 3: The Supreme Auditor**\n\n🔄 *Checking for collusion &"
        " certifying ground truth...*"
    )
    audit_out = run_auditor(
        st.session_state["cloud_file"], user_query, scholar_out, inquisitor_out
    )
    card3.success("⚖️ **Agent 3: The Supreme Auditor**\n\n🏆 *Verdict Certified*")

    st.divider()

    # --- TOP ROW: CERTIFIED VERDICT & INTEGRITY GAUGE ---
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
      st.plotly_chart(
          create_gauge(audit_out.system_integrity_score),
          use_container_width=True,
      )

    st.divider()

    # --- DETAILED REASONING BREAKDOWN TABS ---
    t1, t2, t3 = st.tabs([
        "🧑‍🏫 Scholar Synthesis",
        "⚔️ Inquisitor Challenge",
        "🗂️ Full Audit Log",
    ])

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
      st.progress(
          int(inquisitor_out.chart_reading_accuracy_pct) / 100,
          text=f"Chart Reading Fidelity: {inquisitor_out.chart_reading_accuracy_pct}%",
      )

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
          "agent_3_auditor": audit_out.model_dump(),
      })