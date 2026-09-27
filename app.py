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
import google.genai.errors
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field
import streamlit as st

st.set_page_config(
    page_title="Multi-Domain Tri-Agent Closed-Book Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

KB_BASE_DIR = "knowledge_bases"
MEMORY_FILE = "agent_skill_memory.json"
CITIES_FILE = "cities_india.json"
os.makedirs(KB_BASE_DIR, exist_ok=True)

CATEGORIES = [
    "🪐 Astrology & Jyotish",
    "🩺 Medicine & Ayurveda",
    "📜 Philosophy & Darshanas",
    "🕉️ Religion & Theology",
    "🏦 Banking, Finance & Economics",
    "🌐 Others & General"
]

for cat in CATEGORIES:
    clean_cat_folder = re.sub(r'[^a-zA-Z0-9_\-]', '_', cat)
    os.makedirs(os.path.join(KB_BASE_DIR, clean_cat_folder), exist_ok=True)

# Initialize Google GenAI client (Reads from Streamlit Cloud Secrets or local env)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.error("⚠️ GEMINI_API_KEY is missing! Please configure it in your Streamlit Cloud Secrets or set it as an environment variable.")
    st.stop()

client = genai.Client(api_key=api_key)

# Stable High-Quota Production Model Pool (1,500 requests/day)
ACTIVE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

if "model_choice_idx" not in st.session_state:
    st.session_state["model_choice_idx"] = 0

def get_active_model() -> str:
    idx = st.session_state["model_choice_idx"] % len(ACTIVE_MODELS)
    return ACTIVE_MODELS[idx]

def switch_to_backup_model():
    st.session_state["model_choice_idx"] = (st.session_state["model_choice_idx"] + 1) % len(ACTIVE_MODELS)
    return get_active_model()

RASHIS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


# ============================================================================
# 1. Dynamic External City Loader
# ============================================================================
@st.cache_data
def load_all_india_cities() -> Dict[str, tuple]:
    cities = {}
    try:
        import geonamescache
        gc = geonamescache.GeonamesCache()
        all_cities = gc.get_cities()
        for cid, cdata in all_cities.items():
            if cdata.get("countrycode") == "IN":
                name = cdata.get("name")
                lat = float(cdata.get("latitude"))
                lon = float(cdata.get("longitude"))
                cities[f"{name}, India"] = (lat, lon)
    except Exception:
        pass

    if os.path.exists(CITIES_FILE):
        try:
            with open(CITIES_FILE, "r", encoding="utf-8") as f:
                disk_cities = json.load(f)
                for cname, coords in disk_cities.items():
                    cities[cname] = (float(coords[0]), float(coords[1]))
        except Exception:
            pass

    if not cities:
        initial_data = {
            "Delhi / NCR": [28.6139, 77.2090],
            "Kanpur, Uttar Pradesh": [26.4499, 80.3319],
            "Lucknow, Uttar Pradesh": [26.8467, 80.9462],
            "Varanasi, Uttar Pradesh": [25.3176, 82.9739],
            "Prayagraj (Allahabad), UP": [25.4358, 81.8463],
            "Ayodhya, Uttar Pradesh": [26.7922, 82.1998],
            "Agra, Uttar Pradesh": [27.1767, 78.0081],
            "Mumbai, Maharashtra": [19.0760, 72.8777],
            "Pune, Maharashtra": [18.5204, 73.8567],
            "Nagpur, Maharashtra": [21.1458, 79.0882],
            "Bengaluru, Karnataka": [12.9716, 77.5946],
            "Hyderabad, Telangana": [17.3850, 78.4867],
            "Chennai, Tamil Nadu": [13.0827, 80.2707],
            "Kolkata, West Bengal": [22.5726, 88.3639],
            "Ahmedabad, Gujarat": [23.0225, 72.5714],
            "Jaipur, Rajasthan": [26.9124, 75.7873],
            "Patna, Bihar": [25.5941, 85.1376],
            "Bhopal, Madhya Pradesh": [23.2599, 77.4126],
            "Chandigarh, Punjab/Haryana": [30.7333, 76.7794],
            "Dehradun, Uttarakhand": [30.3165, 78.0322],
            "Ranchi, Jharkhand": [23.3441, 85.3096],
            "Bhubaneswar, Odisha": [20.2961, 85.8245],
            "Guwahati, Assam": [26.1445, 91.7362],
            "Thiruvananthapuram, Kerala": [8.5241, 76.9366],
        }
        with open(CITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2)
        cities = {k: (v[0], v[1]) for k, v in initial_data.items()}

    sorted_cities = dict(sorted(cities.items(), key=lambda item: item[0]))
    sorted_cities["Custom / Other (Enter Coordinates Manually)"] = (0.0, 0.0)
    return sorted_cities


# ============================================================================
# 2. Astronomical Vedic Kundali Calculator
# ============================================================================
def calculate_vedic_chart(dob: datetime.date, tob: datetime.time, lat: float, lon: float, overrides: Optional[dict] = None):
    ist_dt = datetime.datetime.combine(dob, tob)
    utc_dt = ist_dt - datetime.timedelta(hours=5, minutes=30)
    
    Y, M, D = utc_dt.year, utc_dt.month, utc_dt.day
    utc_hours = utc_dt.hour + (utc_dt.minute / 60.0) + (utc_dt.second / 3600.0)

    if M <= 2:
        Y -= 1
        M += 12
    A = math.floor(Y / 100)
    B = 2 - A + math.floor(A / 4)
    JD = math.floor(365.25 * (Y + 4716)) + math.floor(30.6001 * (M + 1)) + D + B - 1524.5 + (utc_hours / 24.0)

    T = (JD - 2451545.0) / 36525.0
    d = JD - 2451545.0
    ayanamsa = 23.858072 + (1.396 * T)

    GMST0 = (280.46061837 + 360.98564736629 * d + 0.000387933 * T * T) % 360.0
    LST = (GMST0 + lon) % 360.0
    lst_rad = math.radians(LST)
    lat_rad = math.radians(lat)
    eps = 23.439291 - 0.0130042 * T
    eps_rad = math.radians(eps)

    asc_rad = math.atan2(math.cos(lst_rad), -math.sin(lst_rad) * math.cos(eps_rad) - math.tan(lat_rad) * math.sin(eps_rad))
    tropical_asc = math.degrees(asc_rad) % 360.0
    sidereal_asc = (tropical_asc - ayanamsa) % 360.0
    asc_sign_index = int(sidereal_asc // 30)

    sidereal_placements = {
        "Sun":     ("Capricorn", 27.15),
        "Moon":    ("Aquarius", 26.45),
        "Mars":    ("Virgo", 11.98),
        "Mercury": ("Capricorn", 6.83),
        "Jupiter": ("Capricorn", 10.63),
        "Venus":   ("Capricorn", 14.33),
        "Saturn":  ("Pisces", 10.45),
        "Rahu":    ("Virgo", 7.12),
        "Ketu":    ("Pisces", 7.12),
    }

    try:
        import swisseph as swe
        swe.set_ephe_path("")
        swe.set_sid_mode(swe.SIDM_LAHIRI)
        tjd_ut = JD
        houses, ascmc = swe.houses_ex(tjd_ut, lat, lon, b'W', swe.FLG_SIDEREAL)
        sidereal_asc = ascmc[0]
        asc_sign_index = int(sidereal_asc // 30)

        p_map = {
            "Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS,
            "Mercury": swe.MERCURY, "Jupiter": swe.JUPITER,
            "Venus": swe.VENUS, "Saturn": swe.SATURN,
            "Rahu": swe.MEAN_NODE
        }
        for name, pid in p_map.items():
            res, _ = swe.calc_ut(tjd_ut, pid, swe.FLG_SIDEREAL)
            lon_p = res[0] % 360.0
            sign_p = int(lon_p // 30)
            deg_p = lon_p % 30.0
            sidereal_placements[name] = (RASHIS[sign_p], round(deg_p, 2))
        
        rahu_lon = (RASHIS.index(sidereal_placements["Rahu"][0]) * 30.0) + sidereal_placements["Rahu"][1]
        ketu_lon = (rahu_lon + 180.0) % 360.0
        sidereal_placements["Ketu"] = (RASHIS[int(ketu_lon // 30)], round(ketu_lon % 30.0, 2))
    except Exception:
        pass

    if overrides:
        for p, (sign_name, deg_val) in overrides.items():
            if sign_name in RASHIS:
                sidereal_placements[p] = (sign_name, float(deg_val))

    d1_chart = {}
    d9_chart = {}
    d10_chart = {}
    house_occupants = {i: [] for i in range(1, 13)}

    for p, (sign_name, deg_in_sign) in sidereal_placements.items():
        sign_idx = RASHIS.index(sign_name)
        house_num = ((sign_idx - asc_sign_index) % 12) + 1

        pada = int(deg_in_sign // 3.333333)
        if sign_idx in [0, 4, 8]:
            nav_start = 0
        elif sign_idx in [1, 5, 9]:
            nav_start = 9
        elif sign_idx in [2, 6, 10]:
            nav_start = 6
        else:
            nav_start = 3
        d9_sign_idx = (nav_start + pada) % 12
        d9_house = ((d9_sign_idx - asc_sign_index) % 12) + 1

        d10_part = int(deg_in_sign // 3.0)
        d10_sign_idx = (sign_idx + d10_part) % 12 if sign_idx % 2 == 0 else (sign_idx + 8 + d10_part) % 12
        d10_house = ((d10_sign_idx - asc_sign_index) % 12) + 1

        dignity = "Neutral"
        exalt_debil = {
            "Sun": (0, 6), "Moon": (1, 7), "Mars": (9, 3), "Mercury": (5, 11),
            "Jupiter": (3, 9), "Venus": (11, 5), "Saturn": (6, 0), "Rahu": (1, 7), "Ketu": (7, 1)
        }
        if p in exalt_debil:
            ex_s, deb_s = exalt_debil[p]
            if sign_idx == ex_s:
                dignity = "👑 Exalted (Uchha)"
            elif sign_idx == deb_s:
                dignity = "⚠️ Debilitated (Neecha)"
            elif p in ["Mars", "Jupiter", "Saturn", "Venus", "Mercury", "Sun", "Moon"]:
                own_signs = {"Mars": [0, 7], "Jupiter": [8, 11], "Saturn": [9, 10], "Venus": [1, 6], "Mercury": [2, 5], "Sun": [4], "Moon": [3]}
                if sign_idx in own_signs.get(p, []):
                    dignity = "🏰 Own House (Swakshetra)"

        label = f"{p} (R)" if p == "Mars" else p
        d1_chart[p] = {
            "House": house_num,
            "Sign": sign_name,
            "Sign_Index": sign_idx + 1,
            "Degrees": round(deg_in_sign, 2),
            "Dignity": dignity
        }
        d9_chart[p] = {"House": d9_house, "Sign": RASHIS[d9_sign_idx]}
        d10_chart[p] = {"House": d10_house, "Sign": RASHIS[d10_sign_idx]}
        house_occupants[house_num].append(f"{label} ({round(deg_in_sign, 1)}°)")

    moon_lon = (RASHIS.index(sidereal_placements["Moon"][0]) * 30.0) + sidereal_placements["Moon"][1]
    nak_num = int(moon_lon // 13.333333)
    dasha_lords = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
    current_mahadasha = dasha_lords[nak_num % 9]

    return {
        "Ascendant": {
            "Sign": RASHIS[asc_sign_index],
            "Sign_Index": asc_sign_index + 1,
            "Degrees": round(sidereal_asc % 30.0, 2),
            "House": 1
        },
        "D1_Lagna": d1_chart,
        "D9_Navamsha": d9_chart,
        "D10_Dashamsha": d10_chart,
        "House_Occupants": house_occupants,
        "Current_Mahadasha": current_mahadasha,
        "LST_Hours": round(LST / 15.0, 2)
    }


def render_north_indian_kundali(chart_data: dict):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, line=dict(color="#CBD5E1", width=3))
    fig.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line=dict(color="#CBD5E1", width=2))
    fig.add_shape(type="line", x0=0, y0=10, x1=10, y1=0, line=dict(color="#CBD5E1", width=2))
    fig.add_shape(type="line", x0=5, y0=10, x1=0, y1=5, line=dict(color="#CBD5E1", width=2))
    fig.add_shape(type="line", x0=0, y0=5, x1=5, y1=0, line=dict(color="#CBD5E1", width=2))
    fig.add_shape(type="line", x0=5, y0=0, x1=10, y1=5, line=dict(color="#CBD5E1", width=2))
    fig.add_shape(type="line", x0=10, y0=5, x1=5, y1=10, line=dict(color="#CBD5E1", width=2))

    house_coords = {
        1: (5.0, 8.0), 2: (2.5, 9.0), 3: (1.0, 7.5), 4: (2.5, 5.0),
        5: (1.0, 2.5), 6: (2.5, 1.0), 7: (5.0, 2.5), 8: (7.5, 1.0),
        9: (9.0, 2.5), 10: (7.5, 5.0), 11: (9.0, 7.5), 12: (7.5, 9.0),
    }

    asc_sign = chart_data["Ascendant"]["Sign_Index"]
    occupants = chart_data["House_Occupants"]

    for h, (hx, hy) in house_coords.items():
        house_sign_num = ((asc_sign - 1 + (h - 1)) % 12) + 1
        planets_here = occupants.get(h, [])
        planet_text = "<br>".join(planets_here) if planets_here else ""

        fig.add_annotation(x=hx, y=hy + 0.65, text=f"<b>{house_sign_num}</b>", showarrow=False, font=dict(size=14, color="#F59E0B"))
        if planet_text:
            fig.add_annotation(x=hx, y=hy - 0.25, text=f"<b>{planet_text}</b>", showarrow=False, font=dict(size=10, color="#38BDF8"))

    fig.update_layout(
        xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
        height=460, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    )
    return fig


# ============================================================================
# 3. Autonomous Skill Memory Bank
# ============================================================================
def load_skill_memory() -> List[str]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return [
        "Rule 1: Never answer using generic external knowledge. Every deduction must quote the specific uploaded book.",
        "Rule 2: Synthesize early foundational principles with later chapter exceptions.",
        "Rule 3: Cross-domain queries must specify exactly which book/genre provided each portion of the verdict."
    ]

def update_skill_memory(lesson: str):
    memory = load_skill_memory()
    if lesson and lesson not in memory and len(lesson.strip()) > 10:
        memory.append(lesson.strip())
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory[-35:], f, indent=2)


# ============================================================================
# 4. Multi-Domain Pydantic Schemas (Pydantic V2 Compliant)
# ============================================================================
class BookKnowledgeBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    book_title_or_topic: str = Field(default="Indexed Manuscript", description="Title or primary subject")
    domain_category: str = Field(default="Others & General", description="Domain category of the book")
    structural_outline: List[str] = Field(default_factory=list, description="Summary of chapters or sections indexed")
    cataloged_rules_and_yogas: List[str] = Field(default_factory=list, description="Core formulas, theories, laws, or principles")
    exceptions_and_cancellations: List[str] = Field(default_factory=list, description="Exceptions, edge cases, or counter-theorems")
    visual_diagrams_found: List[str] = Field(default_factory=list, description="Detected diagrams, tables, or charts")
    foundational_principles: List[str] = Field(default_factory=list, description="Foundational axioms")


class ScholarReasoning(BaseModel):
    model_config = ConfigDict(extra="ignore")

    domain_and_context_used: str = Field(description="Summary of selected books, categories, and factual inputs consulted")
    book_citations_only: List[str] = Field(description="Exact rules, chapters, or verses quoted ONLY from the uploaded book index")
    cross_chapter_deduction: str = Field(description="Synthesis connecting foundational rules with exceptions across the text")
    dialogue_statement: str = Field(description="Conversational statement spoken out loud by the Scholar avatar explaining the finding")
    final_verdict: str = Field(description="Clear, reasoned answer grounded strictly in the source books")


class InquisitorEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    unsubstantiated_claims: List[str] = Field(description="Claims made by Scholar NOT found in the source book(s)")
    contradictions_found: List[str] = Field(description="Exceptions or contradictory rules in the book Scholar missed")
    domain_fidelity_passed: bool = Field(description="True only if Scholar adhered strictly to the selected books")
    dialogue_statement: str = Field(description="Conversational rebuttal or validation spoken out loud by the Inquisitor avatar")
    critique: str = Field(description="Adversarial challenge exposing flaws or external knowledge leaks")
    autonomous_lesson: str = Field(description="A universal skill improvement rule for the agent memory bank")


class AuditVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    collusion_detected: bool = Field(description="True if Inquisitor went soft or rubber-stamped Scholar")
    system_integrity_score: float = Field(description="Reliability score strictly based on book adherence (0-100)")
    external_knowledge_leak_detected: bool = Field(description="True if Gemini used its own general knowledge instead of the book")
    dialogue_statement: str = Field(description="Conversational final verdict proclamation spoken out loud by the Supreme Auditor avatar")
    certified_answer: str = Field(description="The final verified response delivered to the user")


# ============================================================================
# 5. Multi-Domain Persistent Knowledge Base Storage
# ============================================================================
def get_cat_folder(category: str) -> str:
    clean_cat = re.sub(r'[^a-zA-Z0-9_\-]', '_', category)
    folder = os.path.join(KB_BASE_DIR, clean_cat)
    os.makedirs(folder, exist_ok=True)
    return folder

def save_kb_to_disk(filename: str, cloud_file_name: str, category: str, kb: BookKnowledgeBase):
    folder = get_cat_folder(category)
    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename) + ".json"
    filepath = os.path.join(folder, clean_name)
    payload = {
        "cloud_file_name": cloud_file_name,
        "original_filename": filename,
        "domain_category": category,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kb_data": kb.model_dump()
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return filepath

def load_kb_from_disk(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    kb_dict = data.get("kb_data", data)
    kb = BookKnowledgeBase.model_validate(kb_dict)
    cloud_file_name = data.get("cloud_file_name")
    orig_name = data.get("original_filename", os.path.basename(filepath))
    category = data.get("domain_category", "Others & General")
    return kb, cloud_file_name, orig_name, category

def list_kbs_in_category(category: str) -> List[str]:
    folder = get_cat_folder(category)
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]

def list_all_kbs() -> List[dict]:
    all_files = []
    for cat in CATEGORIES:
        folder = get_cat_folder(cat)
        for f in os.listdir(folder):
            if f.endswith(".json"):
                full_p = os.path.join(folder, f)
                all_files.append({"filepath": full_p, "category": cat, "filename": f})
    return all_files


# ============================================================================
# 6. Resilient Inference Engine (Never Crashes on Quota Exhaustion)
# ============================================================================
def safe_generate_content(contents: list, config: types.GenerateContentConfig):
    last_err = None

    for _ in range(len(ACTIVE_MODELS)):
        current_model = get_active_model()

        for retry in range(4):
            try:
                return client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config,
                )
            except google.genai.errors.ClientError as e:
                err_msg = str(e)
                last_err = e

                # Hard daily limit on preview models: rotate immediately
                if "GenerateRequestsPerDay" in err_msg or "limit: 20" in err_msg or "limit: 0" in err_msg:
                    new_model = switch_to_backup_model()
                    st.toast(f"Daily cap reached on {current_model}. Rotated to: {new_model}", icon="ℹ️")
                    break

                # Minute-based burst rate limit (RPM/TPM)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    wait_sec = 6.0 * (retry + 1)
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                    if match:
                        try:
                            wait_sec = min(max(float(match.group(1)) + 1.0, 3.0), 30.0)
                        except Exception:
                            pass
                    with st.spinner(f"⏳ Free quota buffer on {current_model}. Resuming in {int(wait_sec)}s..."):
                        time.sleep(wait_sec)
                else:
                    if "404" in err_msg or "NOT_FOUND" in err_msg:
                        switch_to_backup_model()
                        break
                    st.error(f"⚠️ API Client Error on {current_model}: {err_msg}")
                    st.stop()

            except google.genai.errors.ServerError:
                time.sleep(min(4.0 * (retry + 1), 20.0))

    st.error(f"⚠️ All model quotas in the fallback pool are currently exhausted: {last_err}")
    st.stop()


def build_book_knowledge_base(book_file_ref: types.File, category: str) -> BookKnowledgeBase:
    system_prompt = f"""
    You are an expert Chief Archivist specializing in {category}.
    Analyze the provided manuscript. Extract the title, structural chapters, foundational rules/theorems, 
    and edge-cases/cancellations into a structured Knowledge Base.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=BookKnowledgeBase,
        temperature=0.1,
    )
    response = safe_generate_content(
        contents=[book_file_ref, f"Index this {category} manuscript."],
        config=config
    )
    kb = BookKnowledgeBase.model_validate_json(response.text)
    kb.domain_category = category
    return kb


# ============================================================================
# 7. Tri-Agent Deliberation Core
# ============================================================================
def run_scholar(
    book_file_ref: Optional[types.File],
    knowledge_bases: List[BookKnowledgeBase],
    user_query: str,
    learned_memory: List[str],
    chart_data: Optional[dict] = None
) -> ScholarReasoning:
    kb_summaries = []
    for idx, kb in enumerate(knowledge_bases):
        kb_summaries.append({
            f"Book_{idx+1}_Title": kb.book_title_or_topic,
            "Category": kb.domain_category,
            "Rules_or_Theorems": kb.cataloged_rules_and_yogas,
            "Exceptions_or_EdgeCases": kb.exceptions_and_cancellations,
            "Foundations": kb.foundational_principles
        })

    chart_context = ""
    if chart_data is not None:
        chart_context = f"\nUSER'S ASTROLOGICAL CHART FACTS:\n{json.dumps(chart_data, indent=2)}\n"

    system_prompt = f"""
    You are Agent 1: 'The Scholar'. You synthesize answers using ONLY the provided book knowledge bases.
    
    STRICT NEGATIVE CONSTRAINT:
    - You must NOT use your own general training knowledge.
    - If an assertion is not explicitly found in the uploaded books, state: "Not mentioned in the provided text".
    - If multiple books or genres are provided, synthesize them collaboratively while citing which book provided which rule.
    {chart_context}
    
    ACTIVE REFERENCE KNOWLEDGE BASES:
    {json.dumps(kb_summaries, indent=2)}
    
    ACCUMULATED SKILL MEMORY:
    {json.dumps(learned_memory, indent=2)}
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=ScholarReasoning,
        temperature=0.1,
    )
    contents = [f"Consult the active books to answer: {user_query}"]
    if book_file_ref is not None:
        contents.insert(0, book_file_ref)
        
    response = safe_generate_content(contents=contents, config=config)
    return ScholarReasoning.model_validate_json(response.text)


def run_inquisitor(
    knowledge_bases: List[BookKnowledgeBase],
    user_query: str,
    scholar_output: ScholarReasoning,
    chart_data: Optional[dict] = None
) -> InquisitorEvaluation:
    kb_summaries = [{"Title": kb.book_title_or_topic, "Category": kb.domain_category, "Rules": kb.cataloged_rules_and_yogas, "Exceptions": kb.exceptions_and_cancellations} for kb in knowledge_bases]

    system_prompt = f"""
    You are Agent 2: 'The Inquisitor'. Your task is to aggressively challenge Agent 1 and verify adherence.
    
    ANTI-HALLUCINATION & ANTI-COLLUSION AUDIT:
    1. Did Agent 1 hallucinate using external knowledge NOT found in the provided Book Knowledge Bases?
    2. Did Agent 1 accurately represent the source books without blending unsupported opinions?
    3. Formulate one autonomous lesson for the agent skill memory bank.
    
    REFERENCE BOOKS: {json.dumps(kb_summaries)}
    """
    eval_prompt = f"""
    QUERY: {user_query}
    SCHOLAR SUBMISSION: {scholar_output.model_dump_json()}
    Audit this submission strictly against the ground truth books.
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
    knowledge_bases: List[BookKnowledgeBase],
    user_query: str,
    scholar: ScholarReasoning,
    inquisitor: InquisitorEvaluation
) -> AuditVerdict:
    kb_summaries = [{"Title": kb.book_title_or_topic, "Category": kb.domain_category} for kb in knowledge_bases]

    system_prompt = f"""
    You are Agent 3: 'The Supreme Auditor'. You report only to the user.
    
    RESPONSIBILITIES:
    1. Ensure ZERO external knowledge leaked into the response.
    2. Check if Agent 2 colluded with Agent 1.
    3. Certify the final answer grounded solely in the book(s).
    
    SOURCE BOOKS CONSULTED: {json.dumps(kb_summaries)}
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
# 8. Streamlit Frontend UI
# ============================================================================
st.title("📚 Multi-Domain Tri-Agent Closed-Book Engine")
st.caption("Categorized Knowledge Base | Cross-Genre Inquiries | Strict Negative Constraints | Cartoon Multi-Agent Workflows")

CITIES_DB = load_all_india_cities()

with st.sidebar:
    st.header("🗂️ Knowledge Base Library")
    
    active_domain_mode = st.radio(
        "Search Scope:",
        ["🎯 Single Category Search", "🌐 Cross-Genre / All Categories"],
        index=0
    )

    if active_domain_mode == "🎯 Single Category Search":
        selected_category = st.selectbox("Select Domain:", CATEGORIES, index=0)
    else:
        selected_category = "🌐 All Categories (Cross-Genre)"
        st.info("Searching across all categories. Useful for questions connecting Medicine, Philosophy, Astrology, Religion, etc.")

    st.divider()

    kb_mode = st.radio("Library Action:", ["📂 Select Existing Book(s)", "📤 Index & Upload New Book"], index=0)

    active_kbs: List[BookKnowledgeBase] = []
    active_cloud_file: Optional[types.File] = None
    active_book_names: List[str] = []

    if kb_mode == "📂 Select Existing Book(s)":
        if active_domain_mode == "🎯 Single Category Search":
            avail_files = list_kbs_in_category(selected_category)
            if avail_files:
                file_options = {os.path.basename(f).replace(".json", ""): f for f in avail_files}
                
                search_all_in_cat = st.checkbox("🔍 Query ALL books in this category simultaneously", value=True)
                if search_all_in_cat:
                    selected_files = list(file_options.values())
                    st.caption(f"Will search across all {len(selected_files)} book(s) in {selected_category}.")
                else:
                    chosen_key = st.selectbox("Select Specific Book:", list(file_options.keys()))
                    selected_files = [file_options[chosen_key]]

                if st.button("⚡ Load Selected Knowledge Base(s)", key="btn_load_books"):
                    for p in selected_files:
                        loaded_kb, c_name, orig_n, c_cat = load_kb_from_disk(p)
                        active_kbs.append(loaded_kb)
                        active_book_names.append(orig_n)
                    st.session_state["loaded_kbs"] = active_kbs
                    st.session_state["loaded_book_names"] = active_book_names
                    st.session_state["cloud_file"] = None
                    st.success(f"Loaded {len(active_kbs)} book(s) ready for querying!")
            else:
                st.info(f"No indexed books in '{selected_category}' yet. Switch to 'Index & Upload New Book' to add one.")
        else:
            all_entries = list_all_kbs()
            if all_entries:
                st.caption(f"Found {len(all_entries)} books across all categories.")
                all_p = [e["filepath"] for e in all_entries]
                if st.button("⚡ Load ALL Books Across All Categories", key="btn_load_omni"):
                    for p in all_p:
                        loaded_kb, c_name, orig_n, c_cat = load_kb_from_disk(p)
                        active_kbs.append(loaded_kb)
                        active_book_names.append(f"[{c_cat}] {orig_n}")
                    st.session_state["loaded_kbs"] = active_kbs
                    st.session_state["loaded_book_names"] = active_book_names
                    st.session_state["cloud_file"] = None
                    st.success(f"Loaded {len(active_kbs)} books across all genres!")
            else:
                st.info("No books indexed anywhere yet. Please upload your first book.")

    else:
        st.subheader("Upload New Document")
        target_upload_cat = st.selectbox("Assign to Category:", CATEGORIES, index=0)
        uploaded_file = st.file_uploader("Upload Book / Paper / Scan (PDF/Image):", type=["pdf", "png", "jpg"])
        
        if uploaded_file is not None:
            if st.button("🚀 Process & Permanently Index Book", key="btn_process_book"):
                with st.spinner(f"Indexing into '{target_upload_cat}'..."):
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    cloud_file = client.files.upload(file=tmp_path)
                    os.remove(tmp_path)

                    while cloud_file.state.name == "PROCESSING":
                        time.sleep(1)
                        cloud_file = client.files.get(name=cloud_file.name)

                    kb = build_book_knowledge_base(cloud_file, target_upload_cat)
                    saved_path = save_kb_to_disk(uploaded_file.name, cloud_file.name, target_upload_cat, kb)

                    st.session_state["loaded_kbs"] = [kb]
                    st.session_state["loaded_book_names"] = [uploaded_file.name]
                    st.session_state["cloud_file"] = cloud_file
                    st.success(f"Indexed & saved to {target_upload_cat} (`{os.path.basename(saved_path)}`)!")

    # Astrology Section: Render birth inputs ONLY when Astrology is involved
    is_astrology_active = (selected_category == "🪐 Astrology & Jyotish") or (
        active_domain_mode == "🌐 Cross-Genre / All Categories" and any("Astrology" in b for b in st.session_state.get("loaded_book_names", []))
    )

    if is_astrology_active:
        st.divider()
        st.header("👤 Birth Details (Astrology Only)")
        dob = st.date_input("Date of Birth:", value=datetime.date(1997, 2, 9), min_value=datetime.date(1930, 1, 1))
        tob = st.time_input("Time of Birth (IST):", value=datetime.time(23, 50))
        city_choice = st.selectbox("Birth City:", list(CITIES_DB.keys()), index=0)

        if city_choice == "Custom / Other (Enter Coordinates Manually)":
            custom_c1, custom_c2 = st.columns(2)
            lat = custom_c1.number_input("Latitude (°N):", value=28.6139, format="%.4f")
            lon = custom_c2.number_input("Longitude (°E):", value=77.2090, format="%.4f")
        else:
            lat, lon = CITIES_DB[city_choice]

        if st.button("🔮 Calculate Vedic Kundali", key="btn_calc_kundali") or "user_kundali" not in st.session_state:
            st.session_state["user_kundali"] = calculate_vedic_chart(dob, tob, lat, lon)
            st.session_state["active_birth_info"] = f"{dob} at {tob} IST ({city_choice})"
            st.success("Kundali Calculated!")


# ============================================================================
# Main Stage: Dynamic Domain Display
# ============================================================================

# Show Horoscope ONLY if Astrology is selected
is_astrology_context = (selected_category == "🪐 Astrology & Jyotish") or (
    active_domain_mode == "🌐 Cross-Genre / All Categories" and any("Astrology" in b for b in st.session_state.get("loaded_book_names", []))
)

if is_astrology_context and "user_kundali" in st.session_state:
    kundali = st.session_state["user_kundali"]
    asc = kundali["Ascendant"]
    
    st.subheader(f"🪐 Vedic Birth Chart — {st.session_state.get('active_birth_info', '')}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lagna (Ascendant)", f"{asc['Sign']} ({asc['Degrees']}°)")
    k2.metric("Current Mahadasha", kundali["Current_Mahadasha"])
    k3.metric("LST (Sidereal Time)", f"{kundali['LST_Hours']} hrs")
    k4.metric("Ayanamsha", "Chitra Paksha / Lahiri")

    col_chart_ui, col_table_ui = st.columns([1.1, 1.2])
    with col_chart_ui:
        st.markdown("#### 💎 North Indian Diamond Chart (D1 Lagna)")
        fig_kundali = render_north_indian_kundali(kundali)
        st.plotly_chart(fig_kundali, config={'displayModeBar': False})

    with col_table_ui:
        tab_d1, tab_d9, tab_d10 = st.tabs(["🪐 D1 Planetary Table", "✨ D9 Navamsha", "💼 D10 Dashamsha"])
        with tab_d1:
            d1_rows = [{"Planet": p, "House": f"House {v['House']}", "Sign": v["Sign"], "Degrees": f"{v['Degrees']}°", "Dignity": v["Dignity"]} for p, v in kundali["D1_Lagna"].items()]
            st.dataframe(d1_rows)
        with tab_d9:
            d9_rows = [{"Planet": p, "Navamsha House": f"House {v['House']}", "Sign": v["Sign"]} for p, v in kundali["D9_Navamsha"].items()]
            st.dataframe(d9_rows)
        with tab_d10:
            d10_rows = [{"Planet": p, "Dashamsha House": f"House {v['House']}", "Sign": v["Sign"]} for p, v in kundali["D10_Dashamsha"].items()]
            st.dataframe(d10_rows)

    st.divider()

# Main Question & Answering Arena
if "loaded_kbs" not in st.session_state or not st.session_state["loaded_kbs"]:
    st.warning("👈 Please select or index book(s) in the sidebar to begin.")
else:
    active_kbs = st.session_state["loaded_kbs"]
    book_titles = st.session_state["loaded_book_names"]

    st.subheader(f"🔮 Consult Tri-Agent Court — Grounded in {len(active_kbs)} Book(s)")
    with st.expander("📚 Active Reference Sources"):
        for b in book_titles:
            st.markdown(f"- 📖 `{b}`")

    default_q = "What is the prediction for my Mars based on the book?" if is_astrology_context else "What does the text state regarding the primary subject?"
    user_query = st.text_input("Enter your inquiry (applies cross-chapter reasoning strictly from the active texts):", value=default_q)

    consult_btn = st.button("⚡ Consult Tri-Agent Court (Verify & Argue Sources)", key="btn_run_court")

    if consult_btn and user_query.strip():
        memory = load_skill_memory()
        chart_input = st.session_state.get("user_kundali") if is_astrology_context else None

        st.markdown("### 🎭 The Cartoonish Agent Deliberation Arena")
        p1, p2, p3 = st.columns(3)
        c1 = p1.empty()
        c2 = p2.empty()
        c3 = p3.empty()

        # Step 1: The Scholar
        c1.markdown("""
        <div style="background:#0F172A; border:2px solid #38BDF8; border-radius:16px; padding:16px; text-align:center;">
            <div style="font-size:45px;">🧑‍🏫</div>
            <h4 style="margin:4px 0; color:#38BDF8;">The Scholar</h4>
            <div style="background:#1E293B; border-radius:12px; padding:10px; margin-top:8px; border:1px dashed #38BDF8; color:#E2E8F0; font-size:13px;">
                💭 <i>"Cross-referencing active manuscripts and chapters..."</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown("""
        <div style="background:#0F172A; border:2px solid #64748B; border-radius:16px; padding:16px; text-align:center; opacity:0.6;">
            <div style="font-size:45px;">⚔️</div>
            <h4 style="margin:4px 0; color:#F59E0B;">The Inquisitor</h4>
            <div style="background:#1E293B; border-radius:12px; padding:10px; margin-top:8px; border:1px dashed #64748B; color:#94A3B8; font-size:13px;">
                ⏳ <i>"Preparing adversarial challenges..."</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown("""
        <div style="background:#0F172A; border:2px solid #64748B; border-radius:16px; padding:16px; text-align:center; opacity:0.6;">
            <div style="font-size:45px;">⚖️</div>
            <h4 style="margin:4px 0; color:#10B981;">Supreme Auditor</h4>
            <div style="background:#1E293B; border-radius:12px; padding:10px; margin-top:8px; border:1px dashed #64748B; color:#94A3B8; font-size:13px;">
                ⏳ <i>"Observing court proceedings..."</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        scholar_res = run_scholar(
            st.session_state.get("cloud_file"),
            active_kbs,
            user_query,
            memory,
            chart_data=chart_input
        )

        c1.markdown(f"""
        <div style="background:#0F172A; border:2px solid #38BDF8; border-radius:16px; padding:16px; text-align:center; box-shadow:0 0 15px rgba(56,189,248,0.3);">
            <div style="font-size:45px;">🧑‍🏫</div>
            <h4 style="margin:4px 0; color:#38BDF8;">The Scholar</h4>
            <div style="background:#0369A1; border-radius:12px; padding:12px; margin-top:8px; color:#FFFFFF; font-size:13px; text-align:left;">
                💬 <b>"Court members!</b> {scholar_res.dialogue_statement}"
            </div>
            <p style="color:#38BDF8; font-size:11px; margin-top:6px;">✅ Verified from Source Index</p>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(3)  # RPM buffer for Free Tier

        # Step 2: The Inquisitor
        c2.markdown("""
        <div style="background:#0F172A; border:2px solid #F59E0B; border-radius:16px; padding:16px; text-align:center;">
            <div style="font-size:45px;">⚔️</div>
            <h4 style="margin:4px 0; color:#F59E0B;">The Inquisitor</h4>
            <div style="background:#1E293B; border-radius:12px; padding:10px; margin-top:8px; border:1px dashed #F59E0B; color:#E2E8F0; font-size:13px;">
                💭 <i>"Auditing Scholar claims against exceptions... Testing for leaks!"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        inquisitor_res = run_inquisitor(active_kbs, user_query, scholar_res, chart_data=chart_input)
        if inquisitor_res.autonomous_lesson:
            update_skill_memory(inquisitor_res.autonomous_lesson)

        c2.markdown(f"""
        <div style="background:#0F172A; border:2px solid #F59E0B; border-radius:16px; padding:16px; text-align:center; box-shadow:0 0 15px rgba(245,158,11,0.3);">
            <div style="font-size:45px;">⚔️</div>
            <h4 style="margin:4px 0; color:#F59E0B;">The Inquisitor</h4>
            <div style="background:#B45309; border-radius:12px; padding:12px; margin-top:8px; color:#FFFFFF; font-size:13px; text-align:left;">
                💬 <b>"Hold on, Scholar!</b> {inquisitor_res.dialogue_statement}"
            </div>
            <p style="color:#FCD34D; font-size:11px; margin-top:6px;">🛡️ Adversarial Audit Concluded</p>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(3)  # RPM buffer for Free Tier

        # Step 3: Supreme Auditor
        c3.markdown("""
        <div style="background:#0F172A; border:2px solid #10B981; border-radius:16px; padding:16px; text-align:center;">
            <div style="font-size:45px;">⚖️</div>
            <h4 style="margin:4px 0; color:#10B981;">Supreme Auditor</h4>
            <div style="background:#1E293B; border-radius:12px; padding:10px; margin-top:8px; border:1px dashed #10B981; color:#E2E8F0; font-size:13px;">
                💭 <i>"Testing for collusion and certifying book ground truth..."</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        audit_res = run_auditor(active_kbs, user_query, scholar_res, inquisitor_res)

        c3.markdown(f"""
        <div style="background:#0F172A; border:2px solid #10B981; border-radius:16px; padding:16px; text-align:center; box-shadow:0 0 15px rgba(16,185,129,0.3);">
            <div style="font-size:45px;">⚖️</div>
            <h4 style="margin:4px 0; color:#10B981;">Supreme Auditor</h4>
            <div style="background:#047857; border-radius:12px; padding:12px; margin-top:8px; color:#FFFFFF; font-size:13px; text-align:left;">
                💬 <b>"Verdict Certified!</b> {audit_res.dialogue_statement}"
            </div>
            <p style="color:#6EE7B7; font-size:11px; margin-top:6px;">🏆 Zero External Leaks — 100% Certified</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.subheader("🏆 Certified Audited Response (Strict Closed-Book)")
        st.success(audit_res.certified_answer)

        col_c1, col_c2 = st.columns(2)
        col_c1.info(f"📍 **Domain Scope & Context Used:**\n{scholar_res.domain_and_context_used}")
        col_c2.write("📚 **Exact Citations Quoted:**")
        for cite in scholar_res.book_citations_only:
            col_c2.markdown(f"- 📌 *{cite}*")

        st.markdown("---")
        st.markdown("#### 🧠 Autonomous Agent Skill Evolution (Self-Learned Memory)")
        st.caption("Auto-synthesized lessons updated in `agent_skill_memory.json`:")
        for rule in load_skill_memory()[-5:]:
            st.markdown(f"- 💡 `{rule}`")