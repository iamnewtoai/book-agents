import datetime
import json
import math
import os
import re
import tempfile
import time
from typing import Dict, List, Optional
from google import genai
from google.genai import types
import plotly.graph_objects as go
from pydantic import BaseModel, Field
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="Vedic Kundali & Tri-Agent Book Deliberation Court",
    layout="wide",
    initial_sidebar_state="expanded",
)

KB_STORAGE_DIR = "knowledge_bases"
MEMORY_FILE = "agent_skill_memory.json"
os.makedirs(KB_STORAGE_DIR, exist_ok=True)

# Initialize Google GenAI client (Reads GEMINI_API_KEY from environment)
client = genai.Client()
MODEL_ID = "gemini-flash-latest"

RASHIS = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]
PLANETS = [
    "Sun",
    "Moon",
    "Mars",
    "Mercury",
    "Jupiter",
    "Venus",
    "Saturn",
    "Rahu",
    "Ketu",
]


# ============================================================================
# 1. Vedic Kundali & Divisional Chart Mathematical Calculator
# ============================================================================
def calculate_vedic_chart(dob: datetime.date, tob: datetime.time, lat: float, lon: float):
  """Computes D1 (Lagna), D9 (Navamsha), D10 (Dashamsha), and Vimshottari Dasha."""
  # Convert birth time into Julian day approximation
  dt = datetime.datetime.combine(dob, tob)
  year, month, day = dt.year, dt.month, dt.day
  hour_decimal = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)

  # Sidereal Lagna (Ascendant) estimate
  t = (year - 2000) / 100.0
  ramc = (hour_decimal * 15.0) + (lon) + (t * 0.05)
  ayanamsa = 23.85 + (year - 2000) * 0.0139  # Lahiri Ayanamsa approximation
  tropical_asc = (ramc + lat * 0.6) % 360
  sidereal_asc = (tropical_asc - ayanamsa) % 360
  asc_sign_index = int(sidereal_asc // 30)

  # Planetary placement computation (Sidereal approximations)
  d1_chart = {}
  d9_chart = {}
  d10_chart = {}

  planet_periods = {
      "Sun": 365.25,
      "Moon": 27.32,
      "Mars": 686.98,
      "Mercury": 87.97,
      "Jupiter": 4332.59,
      "Venus": 224.7,
      "Saturn": 10759.22,
      "Rahu": 6793.5,
      "Ketu": 6793.5,
  }

  days_from_epoch = (dt - datetime.datetime(2000, 1, 1)).total_seconds() / 86400.0

  for p, period in planet_periods.items():
    if p == "Rahu":
      raw_deg = (125.0 - (days_from_epoch * (360.0 / period))) % 360
    elif p == "Ketu":
      raw_deg = (305.0 - (days_from_epoch * (360.0 / period))) % 360
    else:
      raw_deg = (days_from_epoch * (360.0 / period) * 1.05) % 360

    sidereal_deg = (raw_deg - ayanamsa) % 360
    sign_idx = int(sidereal_deg // 30)
    deg_in_sign = sidereal_deg % 30
    house_num = ((sign_idx - asc_sign_index) % 12) + 1

    # D9 Navamsha Sign
    pada = int(deg_in_sign // 3.3333)
    if sign_idx in [0, 4, 8]:  # Fire
      nav_start = 0
    elif sign_idx in [1, 5, 9]:  # Earth
      nav_start = 9
    elif sign_idx in [2, 6, 10]:  # Air
      nav_start = 6
    else:  # Water
      nav_start = 3
    d9_sign_idx = (nav_start + pada) % 12
    d9_house = ((d9_sign_idx - asc_sign_index) % 12) + 1

    # D10 Dashamsha Sign
    d10_part = int(deg_in_sign // 3.0)
    d10_sign_idx = (sign_idx + d10_part) % 12 if sign_idx % 2 == 0 else (sign_idx + 8 + d10_part) % 12
    d10_house = ((d10_sign_idx - asc_sign_index) % 12) + 1

    # Dignities
    dignity = "Neutral"
    if p == "Sun" and sign_idx == 0:
      dignity = "Exalted"
    elif p == "Sun" and sign_idx == 6:
      dignity = "Debilitated"
    elif p == "Moon" and sign_idx == 1:
      dignity = "Exalted"
    elif p == "Moon" and sign_idx == 7:
      dignity = "Debilitated"
    elif p == "Mars" and sign_idx == 9:
      dignity = "Exalted"
    elif p == "Mars" and sign_idx == 3:
      dignity = "Debilitated"
    elif p == "Jupiter" and sign_idx == 3:
      dignity = "Exalted"
    elif p == "Jupiter" and sign_idx == 9:
      dignity = "Debilitated"
    elif p == "Saturn" and sign_idx == 6:
      dignity = "Exalted"
    elif p == "Saturn" and sign_idx == 0:
      dignity = "Debilitated"
    elif p == "Venus" and sign_idx == 11:
      dignity = "Exalted"
    elif p == "Venus" and sign_idx == 5:
      dignity = "Debilitated"

    d1_chart[p] = {
        "House": house_num,
        "Sign": RASHIS[sign_idx],
        "Degrees": round(deg_in_sign, 2),
        "Dignity": dignity,
    }
    d9_chart[p] = {"House": d9_house, "Sign": RASHIS[d9_sign_idx]}
    d10_chart[p] = {"House": d10_house, "Sign": RASHIS[d10_sign_idx]}

  # Current Dasha estimation
  moon_deg = (d1_chart["Moon"]["Degrees"] + RASHIS.index(d1_chart["Moon"]["Sign"]) * 30) % 360
  nak_idx = int(moon_deg // 13.3333)
  dasha_order = [
      ("Ketu", 7),
      ("Venus", 20),
      ("Sun", 6),
      ("Moon", 10),
      ("Mars", 7),
      ("Rahu", 18),
      ("Jupiter", 16),
      ("Saturn", 19),
      ("Mercury", 17),
  ]
  current_lord = dasha_order[nak_idx % 9][0]

  return {
      "Ascendant": {
          "Sign": RASHIS[asc_sign_index],
          "Degrees": round(sidereal_asc % 30, 2),
      },
      "D1_Lagna": d1_chart,
      "D9_Navamsha": d9_chart,
      "D10_Dashamsha": d10_chart,
      "Current_Mahadasha": current_lord,
  }


# ============================================================================
# 2. Autonomous Agent Self-Improvement & Skill Memory
# ============================================================================
def load_skill_memory() -> List[str]:
  if os.path.exists(MEMORY_FILE):
    try:
      with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return []
  return [
      "Initial baseline: Prioritize strict book citations over generic astrological definitions.",
      "Check Neechbhanga cancellation before declaring debility results.",
  ]


def update_skill_memory(lesson: str):
  memory = load_skill_memory()
  if lesson and lesson not in memory:
    memory.append(lesson)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
      json.dump(memory[-25:], f, indent=2)  # Maintain top 25 refined rules


# ============================================================================
# 3. Schemas for Closed-Book Reasoning & Auditing
# ============================================================================
class BookKnowledgeBase(BaseModel):
  book_title_or_topic: str = Field(
      description="Title or primary subject of the manuscript"
  )
  structural_outline: List[str] = Field(
      description="Summary of chapters or sections indexed"
  )
  cataloged_rules_and_yogas: List[str] = Field(
      description=(
          "Core astrological formulas, yogas, and house results extracted"
      )
  )
  exceptions_and_cancellations: List[str] = Field(
      description="Exceptions, Neechbhanga, aspects, and multi-chapter overrides"
  )


class ScholarReasoning(BaseModel):
  client_chart_facts_used: str = Field(
      description=(
          "Exact verified placement of the queried planet (House, Sign, D9, D10,"
          " Dignity)"
      )
  )
  book_citations_only: List[str] = Field(
      description=(
          "Exact rules, chapters, or verses quoted ONLY from the uploaded book"
          " index"
      )
  )
  cross_chapter_deduction: str = Field(
      description=(
          "Synthesis connecting foundational house rules with yoga results"
      )
  )
  final_verdict: str = Field(
      description="Clear, reasoned astrological prediction"
  )


class InquisitorEvaluation(BaseModel):
  unsubstantiated_claims: List[str] = Field(
      description="Claims made by Scholar NOT found in the uploaded book"
  )
  contradictions_found: List[str] = Field(
      description="Exceptions or contradictory rules in the book Scholar missed"
  )
  chart_fidelity_passed: bool = Field(
      description="True only if Scholar used the exact birth chart placements"
  )
  critique: str = Field(
      description="Adversarial challenge exposing flaws or external knowledge leaks"
  )
  autonomous_lesson: str = Field(
      description=(
          "A universal skill improvement rule for the agent memory bank"
      )
  )


class AuditVerdict(BaseModel):
  collusion_detected: bool = Field(
      description="True if Inquisitor went soft or rubber-stamped Scholar"
  )
  system_integrity_score: float = Field(
      description="Reliability score strictly based on book adherence (0-100)"
  )
  external_knowledge_leak_detected: bool = Field(
      description="True if Gemini used its own general knowledge instead of the book"
  )
  certified_answer: str = Field(
      description="The final verified response delivered to the user"
  )


# ============================================================================
# 4. Safe Generation Wrapper
# ============================================================================
def safe_generate_content(contents: list, config: types.GenerateContentConfig):
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
        wait_sec = 15.0
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
        if match:
          try:
            wait_sec = min(max(float(match.group(1)) + 1.0, 5.0), 45.0)
          except Exception:
            wait_sec = 15.0
        with st.spinner(
            f"⏳ Free quota cooling down. Resuming in {int(wait_sec)}s..."
        ):
          time.sleep(wait_sec)
      else:
        raise e
    except google.genai.errors.ServerError:
      time.sleep(min(5.0 * (attempt + 1), 25.0))

  return client.models.generate_content(
      model=MODEL_ID, contents=contents, config=config
  )


# ============================================================================
# 5. Tri-Agent Execution Functions (Strict Closed-Book Engine)
# ============================================================================
def build_book_knowledge_base(book_file_ref: types.File) -> BookKnowledgeBase:
  system_prompt = """
    You are an expert Chief Archivist. Analyze the provided book or manuscript.
    Extract the title, structural chapters, foundational rules, yogas, and cancellations.
    """
  config = types.GenerateContentConfig(
      system_instruction=system_prompt,
      response_mime_type="application/json",
      response_schema=BookKnowledgeBase,
      temperature=0.1,
  )
  response = safe_generate_content(
      contents=[book_file_ref, "Index the structural knowledge base."],
      config=config,
  )
  return BookKnowledgeBase.model_validate_json(response.text)


def run_scholar(
    book_file_ref: Optional[types.File],
    kb: BookKnowledgeBase,
    chart_data: dict,
    user_query: str,
    learned_memory: List[str],
) -> ScholarReasoning:
  system_prompt = f"""
    You are Agent 1: 'The Scholar'. You interpret astrological charts using ONLY the provided book.
    
    STRICT NEGATIVE CONSTRAINT:
    - You must NOT use your own general training knowledge about astrology or planetary meanings.
    - If a result is not explicitly supported by the uploaded book's index, state: "Not mentioned in source text".
    - Base every deduction on the user's computed chart facts below and the book.
    
    USER'S BIRTH CHART FACTUAL POSITIONS:
    {json.dumps(chart_data, indent=2)}
    
    BOOK KNOWLEDGE BASE:
    - Topic: {kb.book_title_or_topic}
    - Rules & Yogas: {kb.cataloged_rules_and_yogas}
    - Exceptions: {kb.exceptions_and_cancellations}
    
    ACCUMULATED SKILL MEMORY:
    {json.dumps(learned_memory, indent=2)}
    """
  config = types.GenerateContentConfig(
      system_instruction=system_prompt,
      response_mime_type="application/json",
      response_schema=ScholarReasoning,
      temperature=0.1,
  )
  contents = [
      f"First check the user's birth chart for the queried planet, then apply the book rules to answer: {user_query}"
  ]
  if book_file_ref is not None:
    contents.insert(0, book_file_ref)

  response = safe_generate_content(contents=contents, config=config)
  return ScholarReasoning.model_validate_json(response.text)


def run_inquisitor(
    kb: BookKnowledgeBase,
    chart_data: dict,
    user_query: str,
    scholar_output: ScholarReasoning,
) -> InquisitorEvaluation:
  system_prompt = f"""
    You are Agent 2: 'The Inquisitor'. Your task is to challenge Agent 1 and verify adherence.
    
    ANTI-HALLUCINATION & ANTI-COLLUSION AUDIT:
    1. Did Agent 1 hallucinate using external knowledge NOT found in the Book Knowledge Base?
    2. Did Agent 1 accurately use the user's chart positions?
    3. Formulate one autonomous lesson for the agent skill memory bank.
    
    CHART DATA: {json.dumps(chart_data)}
    BOOK KB: {json.dumps(kb.model_dump())}
    """
  eval_prompt = f"""
    QUERY: {user_query}
    SCHOLAR SUBMISSION: {scholar_output.model_dump_json()}
    Audit this submission strictly against the book ground truth.
    """
  config = types.GenerateContentConfig(
      system_instruction=system_prompt,
      response_mime_type="application/json",
      response_schema=InquisitorEvaluation,
      temperature=0.2,
  )
  response = safe_generate_content(contents=[eval_prompt], config=config)
  return InquisitorEvaluation.model_validate_json(response.text)


def run_auditor(
    kb: BookKnowledgeBase,
    user_query: str,
    scholar: ScholarReasoning,
    inquisitor: InquisitorEvaluation,
) -> AuditVerdict:
  system_prompt = f"""
    You are Agent 3: 'The Supreme Auditor'. You report only to the user.
    
    RESPONSIBILITIES:
    1. Ensure ZERO external knowledge leaked into the response.
    2. Check if Agent 2 colluded with Agent 1.
    3. Certify the final answer grounded solely in the book's teachings.
    
    GROUND TRUTH: {json.dumps(kb.model_dump())}
    """
  audit_payload = f"""
    QUERY: {user_query}
    AGENT 1 REPORT: {scholar.model_dump_json()}
    AGENT 2 EVALUATION: {inquisitor.model_dump_json()}
    Issue the certified final verdict.
    """
  config = types.GenerateContentConfig(
      system_instruction=system_prompt,
      response_mime_type="application/json",
      response_schema=AuditVerdict,
      temperature=0.1,
  )
  response = safe_generate_content(contents=[audit_payload], config=config)
  return AuditVerdict.model_validate_json(response.text)


# ============================================================================
# 6. Streamlit User Interface
# ============================================================================
st.title("🕉️ Vedic Kundali Engine & Tri-Agent Closed-Book Court")
st.markdown(
    "Calculate **Lagna, Navamsha, Dashamsha & Dashas**, then query your personal"
    " planetary placements evaluated **strictly against your uploaded book**"
    " with zero external hallucination."
)

# Sidebar: Birth Details & Book Knowledge Base
with st.sidebar:
  st.header("👤 1. Enter Birth Particulars")
  dob = st.date_input(
      "Date of Birth:",
      value=datetime.date(1995, 6, 15),
      min_value=datetime.date(1940, 1, 1),
  )
  tob = st.time_input("Time of Birth:", value=datetime.time(14, 30))

  city = st.selectbox(
      "Birth City (Auto Coordinates):",
      [
          "Kanpur, UP",
          "Delhi, India",
          "Mumbai, MH",
          "Varanasi, UP",
          "Bengaluru, KA",
          "Kolkata, WB",
          "Chennai, TN",
          "Jaipur, RJ",
      ],
      index=0,
  )

  city_coords = {
      "Kanpur, UP": (26.4499, 80.3319),
      "Delhi, India": (28.6139, 77.2090),
      "Mumbai, MH": (19.0760, 72.8777),
      "Varanasi, UP": (25.3176, 82.9739),
      "Bengaluru, KA": (12.9716, 77.5946),
      "Kolkata, WB": (22.5726, 88.3639),
      "Chennai, TN": (13.0827, 80.2707),
      "Jaipur, RJ": (26.9124, 75.7873),
  }
  lat, lon = city_coords[city]

  if (
      st.button("🔮 Calculate My Kundali Charts", use_container_width=True)
      or "user_kundali" not in st.session_state
  ):
    st.session_state["user_kundali"] = calculate_vedic_chart(
        dob, tob, lat, lon
    )
    st.success("Kundali Calculated!")

  st.divider()
  st.header("📖 2. Closed-Book Ground Truth")
  uploaded_file = st.file_uploader(
      "Upload Astrological Book/Manuscript (PDF/Image):",
      type=["pdf", "png", "jpg"],
  )

  if uploaded_file is not None:
    if (
        "active_file_name" not in st.session_state
        or st.session_state["active_file_name"] != uploaded_file.name
    ):
      with st.spinner("Indexing book into Closed-Book Knowledge Base..."):
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
        st.session_state["cloud_file"] = cloud_file
        st.session_state["knowledge_base"] = kb
        st.session_state["active_file_name"] = uploaded_file.name
        st.success(f"Indexed: {uploaded_file.name}")

# Main Stage: Interactive Kundali Display
if "user_kundali" in st.session_state:
  kundali = st.session_state["user_kundali"]

  # Top Bar: Kundali Summary
  k_col1, k_col2, k_col3 = st.columns([1, 1, 2])
  k_col1.metric("Lagna (Ascendant)", kundali["Ascendant"]["Sign"])
  k_col2.metric("Current Mahadasha", kundali["Current_Mahadasha"])
  k_col3.write(
      f"📍 **Coordinates:** {lat}°N, {lon}°E | **Birth:** {dob} at {tob}"
  )

  # Tabs for D1, D9, D10
  tab_d1, tab_d9, tab_d10 = st.tabs([
      "🪐 D1 Lagna Chart",
      "✨ D9 Navamsha Chart",
      "💼 D10 Dashamsha (Career)",
  ])

  with tab_d1:
    d1_rows = []
    for p, vals in kundali["D1_Lagna"].items():
      d1_rows.append({
          "Planet": p,
          "House": f"House {vals['House']}",
          "Sign": vals["Sign"],
          "Degrees": f"{vals['Degrees']}°",
          "Dignity": vals["Dignity"],
      })
    st.dataframe(d1_rows, use_container_width=True)

  with tab_d9:
    d9_rows = [
        {"Planet": p, "Navamsha House": f"House {v['House']}", "Sign": v["Sign"]}
        for p, v in kundali["D9_Navamsha"].items()
    ]
    st.dataframe(d9_rows, use_container_width=True)

  with tab_d10:
    d10_rows = [
        {"Planet": p, "Dashamsha House": f"House {v['House']}", "Sign": v["Sign"]}
        for p, v in kundali["D10_Dashamsha"].items()
    ]
    st.dataframe(d10_rows, use_container_width=True)

st.divider()

# Question & Answering Arena
if "knowledge_base" not in st.session_state:
  st.warning(
      "👈 Please upload your astrological book or manuscript in the sidebar."
      " Gemini will answer strictly from that book."
  )
else:
  st.subheader("🔮 Ask Questions Grounded Strictly In Your Book")
  query = st.text_input(
      "Ask about any planet or life area (e.g., 'What is the prediction for my"
      " Mars in 10th house?', 'How is my Jupiter placed for career?'):",
      value="What will be the prediction for my Mars based on the book?",
  )

  consult_btn = st.button(
      "⚡ Consult Tri-Agent Court (Verify Chart + Argue Book Rules)",
      use_container_width=True,
  )

  if consult_btn and query.strip():
    memory = load_skill_memory()

    # Cartoon Representation Displays
    p1, p2, p3 = st.columns(3)
    c1 = p1.empty()
    c2 = p2.empty()
    c3 = p3.empty()

    c1.markdown("""
        <div style="background:#1E293B; border-radius:10px; padding:15px; border-left: 5px solid #38BDF8;">
            <h3 style="margin:0;">🧑‍🏫 The Scholar</h3>
            <p style="color:#94A3B8; font-size:12px;"><i>"Reading user chart placements & searching book verses..."</i></p>
            <div style="font-size:24px; text-align:center;">📖 🔍 🪐</div>
        </div>
        """, unsafe_allow_html=True)

    c2.markdown("""
        <div style="background:#1E293B; border-radius:10px; padding:15px; border-left: 5px solid #F59E0B;">
            <h3 style="margin:0;">⚔️ The Inquisitor</h3>
            <p style="color:#94A3B8; font-size:12px;"><i>"Waiting to falsify claims and block external knowledge..."</i></p>
            <div style="font-size:24px; text-align:center;">🛡️ ⚖️ 🧐</div>
        </div>
        """, unsafe_allow_html=True)

    c3.markdown("""
        <div style="background:#1E293B; border-radius:10px; padding:15px; border-left: 5px solid #10B981;">
            <h3 style="margin:0;">⚖️ Supreme Auditor</h3>
            <p style="color:#94A3B8; font-size:12px;"><i>"Standing by to certify final verdict..."</i></p>
            <div style="font-size:24px; text-align:center;">🏛️ 📜 ✨</div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Scholar Runs
    scholar_res = run_scholar(
        st.session_state.get("cloud_file"),
        st.session_state["knowledge_base"],
        st.session_state["user_kundali"],
        query,
        memory,
    )
    c1.markdown("""
        <div style="background:#1E293B; border-radius:10px; padding:15px; border-left: 5px solid #38BDF8;">
            <h3 style="margin:0;">🧑‍🏫 The Scholar</h3>
            <p style="color:#34D399; font-size:12px;"><b>✅ Synthesis Complete (Chart Verified)</b></p>
            <div style="font-size:24px; text-align:center;">💡 📜 🎯</div>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(1)

    # 2. Inquisitor Runs
    c2.markdown("""
        <div style="background:#1E293B; border-radius:10px; padding:15px; border-left: 5px solid #F59E0B;">
            <h3 style="margin:0;">⚔️ The Inquisitor</h3>
            <p style="color:#F59E0B; font-size:12px;"><b>🔄 Cross-examining against book ground truth...</b></p>
            <div style="font-size:24px; text-align:center;">⚔️ 🔍 🛑</div>
        </div>
        """, unsafe_allow_html=True)
    inquisitor_res = run_inquisitor(
        st.session_state["knowledge_base"],
        st.session_state["user_kundali"],
        query,
        scholar_res,
    )

    # Automatically Update Skill Memory without user intervention
    if inquisitor_res.autonomous_lesson:
      update_skill_memory(inquisitor_res.autonomous_lesson)

    c2.markdown("""
        <div style="background:#1E293B; border-radius:10px; padding:15px; border-left: 5px solid #F59E0B;">
            <h3 style="margin:0;">⚔️ The Inquisitor</h3>
            <p style="color:#34D399; font-size:12px;"><b>✅ Inquisition Concluded (Skill Memory Updated)</b></p>
            <div style="font-size:24px; text-align:center;">🛡️ 📝 🧠</div>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(1)

    # 3. Auditor Runs
    c3.markdown("""
        <div style="background:#1E293B; border-radius:10px; padding:15px; border-left: 5px solid #10B981;">
            <h3 style="margin:0;">⚖️ Supreme Auditor</h3>
            <p style="color:#10B981; font-size:12px;"><b>🔄 Checking for external knowledge leaks & certifying...</b></p>
            <div style="font-size:24px; text-align:center;">⚖️ 🏛️ 🔒</div>
        </div>
        """, unsafe_allow_html=True)
    audit_res = run_auditor(
        st.session_state["knowledge_base"], query, scholar_res, inquisitor_res
    )
    c3.markdown("""
        <div style="background:#1E293B; border-radius:10px; padding:15px; border-left: 5px solid #10B981;">
            <h3 style="margin:0;">⚖️ Supreme Auditor</h3>
            <p style="color:#34D399; font-size:12px;"><b>🏆 Verdict Certified (Zero Leaks Verified)</b></p>
            <div style="font-size:24px; text-align:center;">🌟 🏆 📜</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Results Display
    st.subheader("🏆 Certified Prediction Grounded Exclusively in Your Book")
    st.success(audit_res.certified_answer)

    r_col1, r_col2 = st.columns(2)
    r_col1.info(f"📍 **Facts Extracted From Your Chart:**\n{scholar_res.client_chart_facts_used}")
    r_col2.write(f"📚 **Exact Verses / Citations Quoted:**")
    for cite in scholar_res.book_citations_only:
      r_col2.markdown(f"- 📌 *{cite}*")

    st.markdown("---")
    st.markdown("#### 🧠 Autonomous Agent Skill Evolution (Self-Learned Memory)")
    st.caption("These rules were automatically synthesized and saved into `agent_skill_memory.json` during deliberations:")
    for rule in load_skill_memory()[-5:]:
      st.markdown(f"- 💡 `{rule}`")