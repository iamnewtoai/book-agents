import json
import os
import tempfile
import time
from typing import List
from google import genai
from google.genai import types
import plotly.graph_objects as go
from pydantic import BaseModel, Field
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="Tri-Agent Deep Book Reader & Auditor",
    page_layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Google GenAI client (reads GEMINI_API_KEY from environment)
client = genai.Client()
FREE_MODEL = "gemini-2.5-flash"

# ============================================================================
# 1. Pydantic Schemas Enforcing Auditability and Anti-Collusion
# ============================================================================


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
# 2. Agent Execution Functions
# ============================================================================
def run_scholar(
    book_file_ref: types.File, user_query: str
) -> ScholarReasoning:
  system_prompt = """
    You are Agent 1: 'The Scholar'. You have profound expertise in cross-chapter textual analysis, 
    ancient manuscripts, and visual diagrams (astrological Kundalis, tables, ephemerides).
    
    RULES:
    1. Base all deductions ONLY on the provided document.
    2. Synthesize concepts that span early chapters (foundations) and later chapters (specific rules/exceptions).
    3. If an astrological chart or diagram is present, explicitly state house positions, aspects, 
       and planetary dignities (Exalted, Debilitated, Moolatrikona).
    4. Provide strict structural citations.
    """
  response = client.models.generate_content(
      model=FREE_MODEL,
      contents=[
          book_file_ref,
          f"Analyze the source document and answer: {user_query}",
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
    book_file_ref: types.File, user_query: str, scholar_output: ScholarReasoning
) -> InquisitorEvaluation:
  system_prompt = """
    You are Agent 2: 'The Inquisitor'. Your sole mission is to aggressively falsify and challenge Agent 1's work.
    
    ANTI-COLLUSION RULES:
    1. Do NOT assume Agent 1 is correct. Actively hunt for:
       - Misread planetary symbols, house numbers, or degree tables.
       - Selective citation (quoting a general rule while omitting an exception in another chapter).
       - Extrapolations that the text does not strictly justify.
    2. Score visual chart interpretation strictly against the chart images.
    3. You are evaluated on how many actual flaws you uncover.
    """
  eval_prompt = f"""
    USER INQUIRY: {user_query}
    
    AGENT 1'S WORK:
    - Textual Citations: {scholar_output.textual_citations}
    - Visual Observations: {scholar_output.visual_chart_observations}
    - Cross-Chapter Synthesis: {scholar_output.cross_chapter_synthesis}
    - Final Verdict: {scholar_output.final_verdict}
    
    Verify every claim directly against the attached document. Expose all discrepancies.
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
    You are Agent 3: 'The Supreme Auditor'. You answer only to the user.
    
    RESPONSIBILITIES:
    1. Detect collusion: Check whether Agent 2 gave a pass to flawed claims or failed to inspect the chart.
    2. Review the original scanned book independently to determine ground truth.
    3. Correct errors and provide the definitive certified answer with an integrity score (0-100).
    """
  audit_payload = f"""
    ORIGINAL QUERY: {user_query}
    
    AGENT 1 OUTPUT:
    {scholar.model_dump_json(indent=2)}
    
    AGENT 2 EVALUATION:
    {inquisitor.model_dump_json(indent=2)}
    
    Review the source document, check both agents for accuracy and collusion, and issue the final verdict.
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
# 3. Helper: Integrity Gauge Chart
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
# 4. Streamlit User Interface
# ============================================================================
st.title("🏛️ Tri-Agent Deep Book Reader & Adversarial Auditor")
st.markdown(
    "Multi-agent reasoning engine with **Visual Chart/Table Parsing**, **Adversarial"
    " Testing**, and **Double-Blind Anti-Collusion Auditing**."
)

# Sidebar for Book Upload & Management
with st.sidebar:
  st.header("📖 Document Upload")
  uploaded_file = st.file_uploader(
      "Upload Scanned Book / PDF / Chart:", type=["pdf", "png", "jpg", "jpeg"]
  )

  if uploaded_file is not None:
    if (
        "current_file_name" not in st.session_state
        or st.session_state["current_file_name"] != uploaded_file.name
    ):
      with st.spinner("Uploading and indexing visual document with Gemini..."):
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
          tmp.write(uploaded_file.read())
          tmp_path = tmp.name

        cloud_file = client.files.upload(file=tmp_path)
        os.remove(tmp_path)

        while cloud_file.state.name == "PROCESSING":
          time.sleep(1)
          cloud_file = client.files.get(name=cloud_file.name)

        st.session_state["cloud_file"] = cloud_file
        st.session_state["current_file_name"] = uploaded_file.name
        st.success(f"Loaded: {uploaded_file.name}")

  st.divider()
  st.markdown("### 🤖 Agent Architecture")
  st.markdown("""
    - **Agent 1: The Scholar** (Synthesizes cross-chapter rules, ephemerides & Kundalis).
    - **Agent 2: The Inquisitor** (Adversary looking for contradictions & misread charts).
    - **Agent 3: The Auditor** (Validates ground truth & intercepts collusion).
    """)

# Main Interactive Arena
if "cloud_file" not in st.session_state:
  st.info(
      "👈 Please upload a scanned book, chart image, or PDF in the sidebar to"
      " begin."
  )
else:
  st.write(f"📁 **Active Book:** `{st.session_state['current_file_name']}`")

  user_query = st.text_area(
      "Enter your question (e.g. asking about specific charts, cross-chapter rules, yogas):",
      placeholder=(
          "Ex: Based on the chart on page 15 and rules across Chapter 3 and 11,"
          " what is the status of the 10th lord and does any Neechbhanga yoga"
          " form?"
      ),
      height=90,
  )

  ask_button = st.button(
      "🚀 Deliberate with Tri-Agent Court", use_container_width=True
  )

  if ask_button and user_query.strip():
    # Visual Progress Arena
    progress_col1, progress_col2, progress_col3 = st.columns(3)

    card1 = progress_col1.empty()
    card2 = progress_col2.empty()
    card3 = progress_col3.empty()

    # Initial Agent Card Display
    card1.info("🧑‍🏫 **Agent 1: The Scholar**\n\n*Waiting in queue...*")
    card2.info("⚔️ **Agent 2: The Inquisitor**\n\n*Waiting in queue...*")
    card3.info("⚖️ **Agent 3: The Supreme Auditor**\n\n*Waiting in queue...*")

    # Step 1: Agent 1 Reasoning
    card1.warning(
        "🧑‍🏫 **Agent 1: The Scholar**\n\n🔄 *Reading charts, tables &"
        " synthesizing cross-chapter rules...*"
    )
    scholar_out = run_scholar(st.session_state["cloud_file"], user_query)
    card1.success("🧑‍🏫 **Agent 1: The Scholar**\n\n✅ *Synthesis Complete*")

    time.sleep(1)

    # Step 2: Agent 2 Inquisition
    card2.warning(
        "⚔️ **Agent 2: The Inquisitor**\n\n🔄 *Falsifying claims, hunting"
        " contradictions & checking chart...*"
    )
    inquisitor_out = run_inquisitor(
        st.session_state["cloud_file"], user_query, scholar_out
    )
    if inquisitor_out.passed_inquisition:
      card2.success(
          "⚔️ **Agent 2: The Inquisitor**\n\n✅ *Passed Rigorous Inquisition*"
      )
    else:
      card2.error(
          "⚔️ **Agent 2: The Inquisitor**\n\n⚠️ *Discrepancies & Flaws"
          " Detected!*"
      )

    time.sleep(1)

    # Step 3: Agent 3 Audit
    card3.warning(
        "⚖️ **Agent 3: The Supreme Auditor**\n\n🔄 *Checking for collusion &"
        " locking ground truth...*"
    )
    audit_out = run_auditor(
        st.session_state["cloud_file"], user_query, scholar_out, inquisitor_out
    )
    card3.success("⚖️ **Agent 3: The Supreme Auditor**\n\n🏆 *Verdict Certified*")

    st.divider()

    # --- TOP ROW: AUDITED VERDICT & INTEGRITY GAUGES ---
    top_left, top_right = st.columns([2, 1])

    with top_left:
      st.subheader("🏆 Certified Audited Response")
      st.success(audit_out.certified_answer)

      c_col1, c_col2 = st.columns(2)
      if audit_out.collusion_detected:
        c_col1.error("🚨 **Collusion Status:** Collusion / Rubber-Stamping Caught")
      else:
        c_col1.success(
            "🛡️ **Collusion Status:** Verified Independent (Zero Collusion)"
        )

      c_col2.info(f"**Ground Truth:** {audit_out.ground_truth_alignment}")

    with top_right:
      st.plotly_chart(
          create_gauge(audit_out.system_integrity_score),
          use_container_width=True,
      )

    st.divider()

    # --- DETAILED DELIBERATION TABS ---
    tab1, tab2, tab3 = st.tabs([
        "🧑‍🏫 Agent 1: Scholar Synthesis",
        "⚔️ Agent 2: Inquisitor Challenge",
        "📄 Raw Audit Logs",
    ])

    with tab1:
      st.markdown("#### 🔍 Visual Chart & Diagram Extractions")
      if scholar_out.visual_chart_observations:
        for obs in scholar_out.visual_chart_observations:
          st.markdown(f"- `{obs}`")
      else:
        st.write("No diagrams observed.")

      st.markdown("#### 📚 Cross-Chapter Synthesis")
      st.write(scholar_out.cross_chapter_synthesis)

      st.markdown("#### 📑 Exact Source Citations")
      for cite in scholar_out.textual_citations:
        st.markdown(f"- 📌 *{cite}*")

    with tab2:
      st.markdown("#### 🎯 Visual Diagram Reading Fidelity")
      st.progress(
          int(inquisitor_out.chart_reading_accuracy_pct) / 100,
          text=f"Chart Reading Accuracy: {inquisitor_out.chart_reading_accuracy_pct}%",
      )

      st.markdown("#### 🥊 Adversarial Critique")
      st.write(inquisitor_out.adversarial_critique)

      col_f1, col_f2 = st.columns(2)
      with col_f1:
        st.markdown("**Unsubstantiated Assertions:**")
        if inquisitor_out.unsubstantiated_claims:
          for claim in inquisitor_out.unsubstantiated_claims:
            st.error(f"❌ {claim}")
        else:
          st.success("None detected.")

      with col_f2:
        st.markdown("**Contradictions / Overlooked Rules:**")
        if inquisitor_out.contradictions_found:
          for contra in inquisitor_out.contradictions_found:
            st.warning(f"⚠️ {contra}")
        else:
          st.success("None detected.")

    with tab3:
      st.json({
          "agent_1_scholar": scholar_out.model_dump(),
          "agent_2_inquisitor": inquisitor_out.model_dump(),
          "agent_3_auditor": audit_out.model_dump(),
      })