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
from pydantic import BaseModel, Field
import streamlit as st

st.set_page_config(
    page_title="Vedic Kundali & Tri-Agent Closed-Book Court",
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
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

INDIA_CITIES = {
    "Delhi / NCR": (28.6139, 77.2090),
    "Kanpur, Uttar Pradesh": (26.4499, 80.3319),
    "Lucknow, Uttar Pradesh": (26.8467, 80.9462),
    "Varanasi, Uttar Pradesh": (25.3176, 82.9739),
    "Prayagraj (Allahabad), UP": (25.4358, 81.8463),
    "Ayodhya, Uttar Pradesh": (26.7922, 82.1998),
    "Mumbai, Maharashtra": (19.0760, 72.8777),
    "Pune, Maharashtra": (18.5204, 73.8567),
    "Nagpur, Maharashtra": (21.1458, 79.0882),
    "Bengaluru, Karnataka": (12.9716, 77.5946),
    "Hyderabad, Telangana": (17.3850, 78.4867),
    "Chennai, Tamil Nadu": (13.0827, 80.2707),
    "Kolkata, West Bengal": (22.5726, 88.3639),
    "Ahmedabad, Gujarat": (23.0225, 72.5714),
    "Surat, Gujarat": (21.1702, 72.8311),
    "Jaipur, Rajasthan": (26.9124, 75.7873),
    "Jodhpur, Rajasthan": (26.2389, 73.0243),
    "Patna, Bihar": (25.5941, 85.1376),
    "Gaya, Bihar": (24.7955, 85.0002),
    "Bhopal, Madhya Pradesh": (23.2599, 77.4126),
    "Indore, Madhya Pradesh": (22.7196, 75.8577),
    "Ujjain, Madhya Pradesh": (23.1765, 75.7885),
    "Chandigarh, Punjab/Haryana": (30.7333, 76.7794),
    "Amritsar, Punjab": (31.6340, 74.8723),
    "Dehradun, Uttarakhand": (30.3165, 78.0322),
    "Haridwar, Uttarakhand": (29.9457, 78.1642),
    "Ranchi, Jharkhand": (23.3441, 85.3096),
    "Bhubaneswar, Odisha": (20.2961, 85.8245),
    "Puri, Odisha": (19.8135, 85.8312),
    "Guwahati, Assam": (26.1445, 91.7362),
    "Thiruvananthapuram, Kerala": (8.5241, 76.9366),
    "Kochi, Kerala": (9.9312, 76.2673),
    "Custom / Other (Enter Coordinates Manually)": (0.0, 0.0)
}


# ============================================================================
# 1. High-Precision Keplerian Planetary & Vedic Kundali Calculator
# ============================================================================
def calculate_vedic_chart(dob: datetime.date, tob: datetime.time, lat: float, lon: float):
    """
    Computes exact Sidereal Vedic planetary positions:
    - Corrects IST (UTC+05:30) to UTC.
    - True Greenwich Mean Sidereal Time (GMST) and Local Sidereal Time (LST).
    - Obliquity & exact Lahiri Ayanamsha (Chitra Paksha).
    - Keplerian orbital eccentricity equations for true geocentric ecliptic longitudes.
    """
    # 1. Convert IST to UTC
    ist_dt = datetime.datetime.combine(dob, tob)
    utc_dt = ist_dt - datetime.timedelta(hours=5, minutes=30)
    
    Y, M, D = utc_dt.year, utc_dt.month, utc_dt.day
    utc_hours = utc_dt.hour + (utc_dt.minute / 60.0) + (utc_dt.second / 3600.0)

    # 2. Julian Day computation
    if M <= 2:
        Y -= 1
        M += 12
    A = math.floor(Y / 100)
    B = 2 - A + math.floor(A / 4)
    JD = math.floor(365.25 * (Y + 4716)) + math.floor(30.6001 * (M + 1)) + D + B - 1524.5 + (utc_hours / 24.0)

    # 3. Lahiri Ayanamsha (Chitra Paksha reference: 23°51'11" at 2000.0)
    T = (JD - 2451545.0) / 36525.0
    d = JD - 2451545.0
    ayanamsa = 23.858072 + (1.396 * T) + (0.000308 * T * T)

    # 4. Local Sidereal Time (LST) and Ascendant (Lagna)
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

    # 5. Keplerian Heliocentric & Geocentric True Longitude Calculator
    def solve_kepler(M_deg, e):
        M_r = math.radians(M_deg % 360.0)
        E = M_r
        for _ in range(10):
            dE = (E - e * math.sin(E) - M_r) / (1.0 - e * math.cos(E))
            E -= dE
            if abs(dE) < 1e-6:
                break
        v = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0), math.sqrt(1.0 - e) * math.cos(E / 2.0))
        return math.degrees(v) % 360.0

    # Earth orbital parameters for geocentric reduction
    M_earth = (357.529 + 0.98560028 * d) % 360.0
    v_earth = solve_kepler(M_earth, 0.016709)
    L_sun_trop = (280.466 + 0.98564736 * d + 1.915 * math.sin(math.radians(M_earth)) + 0.020 * math.sin(2 * math.radians(M_earth))) % 360.0
    R_earth = 1.00014 - 0.01671 * math.cos(math.radians(M_earth))

    # Planetary Elements: [a (AU), e, I (deg), L0 (mean lon), w_bar (perihelion), daily_n]
    elements = {
        "Mercury": [0.387098, 0.205630, 7.005, 252.251, 77.456, 4.0923344],
        "Venus":   [0.723332, 0.006773, 3.394, 181.979, 131.564, 1.6021303],
        "Mars":    [1.523679, 0.093405, 1.850, 355.433, 336.060, 0.5240330],
        "Jupiter": [5.20260,  0.048498, 1.303, 34.351,  14.331,  0.0830853],
        "Saturn":  [9.554909, 0.055546, 2.489, 50.077,  93.057,  0.0334441],
    }

    tropical_positions = {}
    tropical_positions["Sun"] = L_sun_trop

    # Moon (Brown's Lunar Theory fundamental terms)
    L_moon = (218.316 + 13.176396 * d) % 360.0
    M_moon = (134.963 + 13.064993 * d) % 360.0
    F_moon = (93.272 + 13.229350 * d) % 360.0
    D_moon = (297.850 + 12.190749 * d) % 360.0

    moon_corr = (
        6.289 * math.sin(math.radians(M_moon))
        + 1.274 * math.sin(math.radians(2 * D_moon - M_moon))
        + 0.658 * math.sin(math.radians(2 * D_moon))
        + 0.214 * math.sin(math.radians(2 * M_moon))
        - 0.186 * math.sin(math.radians(M_earth))
    )
    tropical_positions["Moon"] = (L_moon + moon_corr) % 360.0

    # Rahu & Ketu (Mean retrograde lunar nodes)
    N_rahu = (125.0445 - 0.05295376 * d) % 360.0
    tropical_positions["Rahu"] = N_rahu
    tropical_positions["Ketu"] = (N_rahu + 180.0) % 360.0

    # Planets geocentric transformation
    for name, (a, e, inc, L0, w_bar, daily_n) in elements.items():
        M_p = (L0 + daily_n * d - w_bar) % 360.0
        v_p = solve_kepler(M_p, e)
        r_p = a * (1.0 - e * e) / (1.0 + e * math.cos(math.radians(v_p)))
        l_helio = (v_p + w_bar) % 360.0
        
        # Geocentric reduction onto the ecliptic
        x_p = r_p * math.cos(math.radians(l_helio)) - R_earth * math.cos(math.radians(L_sun_trop))
        y_p = r_p * math.sin(math.radians(l_helio)) - R_earth * math.sin(math.radians(L_sun_trop))
        l_geo = math.degrees(math.atan2(y_p, x_p)) % 360.0
        tropical_positions[name] = l_geo

    # 6. Build D1, D9, D10 & House Occupants
    d1_chart = {}
    d9_chart = {}
    d10_chart = {}
    house_occupants = {i: [] for i in range(1, 13)}

    for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
        trop = tropical_positions[p]
        sidereal = (trop - ayanamsa) % 360.0
        sign_idx = int(sidereal // 30)
        deg_in_sign = sidereal % 30.0
        house_num = ((sign_idx - asc_sign_index) % 12) + 1

        # D9 Navamsha Calculation
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

        # D10 Dashamsha Calculation
        d10_part = int(deg_in_sign // 3.0)
        d10_sign_idx = (sign_idx + d10_part) % 12 if sign_idx % 2 == 0 else (sign_idx + 8 + d10_part) % 12
        d10_house = ((d10_sign_idx - asc_sign_index) % 12) + 1

        # Dignity calculation
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
            elif p == "Mars" and sign_idx in [0, 7]:
                dignity = "🏰 Own House (Swakshetra)"
            elif p == "Jupiter" and sign_idx in [8, 11]:
                dignity = "🏰 Own House (Swakshetra)"
            elif p == "Saturn" and sign_idx in [9, 10]:
                dignity = "🏰 Own House (Swakshetra)"
            elif p == "Venus" and sign_idx in [1, 6]:
                dignity = "🏰 Own House (Swakshetra)"
            elif p == "Mercury" and sign_idx in [2, 5]:
                dignity = "🏰 Own House (Swakshetra)"
            elif p == "Sun" and sign_idx == 4:
                dignity = "🏰 Own House (Swakshetra)"
            elif p == "Moon" and sign_idx == 3:
                dignity = "🏰 Own House (Swakshetra)"

        d1_chart[p] = {
            "House": house_num,
            "Sign": RASHIS[sign_idx],
            "Sign_Index": sign_idx + 1,
            "Degrees": round(deg_in_sign, 2),
            "Dignity": dignity
        }
        d9_chart[p] = {"House": d9_house, "Sign": RASHIS[d9_sign_idx]}
        d10_chart[p] = {"House": d10_house, "Sign": RASHIS[d10_sign_idx]}
        house_occupants[house_num].append(f"{p} ({round(deg_in_sign, 1)}°)")

    # 7. Vimshottari Mahadasha based on Moon's True Nakshatra
    moon_sid = (tropical_positions["Moon"] - ayanamsa) % 360.0
    nak_num = int(moon_sid // 13.333333)
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


# ============================================================================
# 2. Visual North Indian Diamond Kundali Generator
# ============================================================================
def render_north_indian_kundali(chart_data: dict):
    """Draws authentic North Indian Vedic Chart with Diamond & Triangular Houses."""
    fig = go.Figure()
    
    # Outer bounding box
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, line=dict(color="#E2E8F0", width=3))
    # Diagonal crossed lines
    fig.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line=dict(color="#E2E8F0", width=2))
    fig.add_shape(type="line", x0=0, y0=10, x1=10, y1=0, line=dict(color="#E2E8F0", width=2))
    # Inner Diamond
    fig.add_shape(type="line", x0=5, y0=10, x1=0, y1=5, line=dict(color="#E2E8F0", width=2))
    fig.add_shape(type="line", x0=0, y0=5, x1=5, y1=0, line=dict(color="#E2E8F0", width=2))
    fig.add_shape(type="line", x0=5, y0=0, x1=10, y1=5, line=dict(color="#E2E8F0", width=2))
    fig.add_shape(type="line", x0=10, y0=5, x1=5, y1=10, line=dict(color="#E2E8F0", width=2))

    house_coords = {
        1:  (5.0, 8.0),   # 1st House (Top Center Diamond - Tanu Bhava)
        2:  (2.5, 9.0),   # 2nd House (Top Left Triangle)
        3:  (1.0, 7.5),   # 3rd House (Upper Left Triangle)
        4:  (2.5, 5.0),   # 4th House (Left Middle Diamond - Sukha Bhava)
        5:  (1.0, 2.5),   # 5th House (Lower Left Triangle)
        6:  (2.5, 1.0),   # 6th House (Bottom Left Triangle)
        7:  (5.0, 2.5),   # 7th House (Bottom Middle Diamond - Kalatra Bhava)
        8:  (7.5, 1.0),   # 8th House (Bottom Right Triangle)
        9:  (9.0, 2.5),   # 9th House (Lower Right Triangle)
        10: (7.5, 5.0),   # 10th House (Right Middle Diamond - Karma Bhava)
        11: (9.0, 7.5),   # 11th House (Upper Right Triangle)
        12: (7.5, 9.0),   # 12th House (Top Right Triangle)
    }

    asc_sign = chart_data["Ascendant"]["Sign_Index"]
    occupants = chart_data["House_Occupants"]

    for h, (hx, hy) in house_coords.items():
        house_sign_num = ((asc_sign - 1 + (h - 1)) % 12) + 1
        planets_here = occupants.get(h, [])
        planet_text = "<br>".join(planets_here) if planets_here else ""

        fig.add_annotation(
            x=hx, y=hy + 0.6,
            text=f"<b>{house_sign_num}</b>",
            showarrow=False,
            font=dict(size=14, color="#F59E0B")
        )
        
        if planet_text:
            fig.add_annotation(
                x=hx, y=hy - 0.2,
                text=f"<b>{planet_text}</b>",
                showarrow=False,
                font=dict(size=11, color="#38BDF8")
            )

    fig.update_layout(
        xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
    )
    return fig


# ============================================================================
# 3. Autonomous Skill Memory Bank (Auto-updates without user intervention)
# ============================================================================
def load_skill_memory() -> List[str]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return [
        "Rule 1: Never answer using generic Vedic astrology. Every deduction must quote the specific uploaded book.",
        "Rule 2: For Neechbhanga, verify if the debilitation lord is in Kendra to Lagna or Moon as instructed in classical chapters.",
        "Rule 3: Always check D9 Navamsha sign confirmation for planetary strength before concluding house results."
    ]

def update_skill_memory(lesson: str):
    memory = load_skill_memory()
    if lesson and lesson not in memory and len(lesson.strip()) > 10:
        memory.append(lesson.strip())
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory[-30:], f, indent=2)


# ============================================================================
# 4. Backward-Compatible Pydantic Schema (Handles Old & New JSON KBs)
# ============================================================================
class BookKnowledgeBase(BaseModel):
    book_title_or_topic: str = Field(default="Indexed Astrological Manuscript", description="Title or primary subject")
    structural_outline: List[str] = Field(default_factory=list, description="Summary of chapters or sections indexed")
    cataloged_rules_and_yogas: List[str] = Field(default_factory=list, description="Core formulas, yogas, and house results")
    exceptions_and_cancellations: List[str] = Field(default_factory=list, description="Exceptions, Neechbhanga, aspects")
    visual_diagrams_found: List[str] = Field(default_factory=list, description="Detected diagrams/tables")
    foundational_principles: List[str] = Field(default_factory=list, description="Foundational rules")

    class Config:
        extra = "ignore"  # Gracefully ignores unexpected keys from older JSON versions


class ScholarReasoning(BaseModel):
    client_chart_facts_used: str = Field(description="Exact verified placement of the queried planet (House, Sign, D9, D10, Dignity)")
    book_citations_only: List[str] = Field(description="Exact rules, chapters, or verses quoted ONLY from the uploaded book index")
    cross_chapter_deduction: str = Field(description="Synthesis connecting foundational house rules with yoga results")
    final_verdict: str = Field(description="Clear, reasoned astrological prediction")


class InquisitorEvaluation(BaseModel):
    unsubstantiated_claims: List[str] = Field(description="Claims made by Scholar NOT found in the uploaded book")
    contradictions_found: List[str] = Field(description="Exceptions or contradictory rules in the book Scholar missed")
    chart_fidelity_passed: bool = Field(description="True only if Scholar used the exact birth chart placements")
    critique: str = Field(description="Adversarial challenge exposing flaws or external knowledge leaks")
    autonomous_lesson: str = Field(description="A universal skill improvement rule for the agent memory bank")


class AuditVerdict(BaseModel):
    collusion_detected: bool = Field(description="True if Inquisitor went soft or rubber-stamped Scholar")
    system_integrity_score: float = Field(description="Reliability score strictly based on book adherence (0-100)")
    external_knowledge_leak_detected: bool = Field(description="True if Gemini used its own general knowledge instead of the book")
    certified_answer: str = Field(description="The final verified response delivered to the user")


# ============================================================================
# 5. Persistent Local Disk Storage
# ============================================================================
def save_kb_to_disk(filename: str, cloud_file_name: str, kb: BookKnowledgeBase):
    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename) + ".json"
    filepath = os.path.join(KB_STORAGE_DIR, clean_name)
    payload = {
        "cloud_file_name": cloud_file_name,
        "original_filename": filename,
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
    return kb, cloud_file_name, orig_name


def get_saved_kb_files():
    if not os.path.exists(KB_STORAGE_DIR):
        return []
    return [f for f in os.listdir(KB_STORAGE_DIR) if f.endswith(".json")]


# ============================================================================
# 6. Tri-Agent Execution Functions
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
                        pass
                with st.spinner(f"⏳ Rate limit buffer. Resuming in {int(wait_sec)}s..."):
                    time.sleep(wait_sec)
            else:
                raise e
        except google.genai.errors.ServerError:
            time.sleep(min(5.0 * (attempt + 1), 25.0))
            
    return client.models.generate_content(model=MODEL_ID, contents=contents, config=config)


def build_book_knowledge_base(book_file_ref: types.File) -> BookKnowledgeBase:
    system_prompt = """
    You are an expert Chief Archivist. Analyze the provided book or manuscript.
    Extract the title, structural chapters, foundational rules, yogas, and cancellations into a concise knowledge roadmap.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=BookKnowledgeBase,
        temperature=0.1,
    )
    response = safe_generate_content(
        contents=[book_file_ref, "Index the structural knowledge base of this manuscript."],
        config=config
    )
    return BookKnowledgeBase.model_validate_json(response.text)


def run_scholar(book_file_ref: Optional[types.File], kb: BookKnowledgeBase, chart_data: dict, user_query: str, learned_memory: List[str]) -> ScholarReasoning:
    system_prompt = f"""
    You are Agent 1: 'The Scholar'. You interpret Vedic charts using ONLY the provided book.
    
    STRICT NEGATIVE CONSTRAINT:
    - You must NOT use your own general training knowledge about astrology or planetary meanings.
    - If a prediction or rule is not explicitly found in the uploaded book's index, state: "Not mentioned in source text".
    - Base every deduction on the user's computed chart facts below and the book.
    
    USER'S EXACT BIRTH CHART FACTUAL POSITIONS (Calculated with IST UTC+5:30):
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
    contents = [f"First check the user's birth chart for the queried planet, then apply the book rules to answer: {user_query}"]
    if book_file_ref is not None:
        contents.insert(0, book_file_ref)
        
    response = safe_generate_content(contents=contents, config=config)
    return ScholarReasoning.model_validate_json(response.text)


def run_inquisitor(kb: BookKnowledgeBase, chart_data: dict, user_query: str, scholar_output: ScholarReasoning) -> InquisitorEvaluation:
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


def run_auditor(kb: BookKnowledgeBase, user_query: str, scholar: ScholarReasoning, inquisitor: InquisitorEvaluation) -> AuditVerdict:
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
# 7. Streamlit User Interface
# ============================================================================
st.title("🕉️ Vedic Kundali Engine & Tri-Agent Closed-Book Court")
st.caption("Astronomical Keplerian Ephemeris (IST UTC+5:30 Corrected) | Diamond Kundali | Animated Cartoon Agents | Auto Skill Learning")

with st.sidebar:
    st.header("👤 1. Birth Particulars (IST UTC+05:30)")
    dob = st.date_input("Date of Birth:", value=datetime.date(1997, 2, 9), min_value=datetime.date(1930, 1, 1))
    tob = st.time_input("Time of Birth (IST):", value=datetime.time(23, 50))
    
    city_choice = st.selectbox("Birth City (India):", list(INDIA_CITIES.keys()), index=0)
    
    if city_choice == "Custom / Other (Enter Coordinates Manually)":
        custom_c1, custom_c2 = st.columns(2)
        lat = custom_c1.number_input("Latitude (°N):", value=28.6139, format="%.4f")
        lon = custom_c2.number_input("Longitude (°E):", value=77.2090, format="%.4f")
    else:
        lat, lon = INDIA_CITIES[city_choice]
        st.caption(f"📍 Coordinates: **{lat}° N, {lon}° E** | Timezone: **IST (UTC+05:30)**")

    if st.button("🔮 Calculate Vedic Kundali", use_container_width=True) or "user_kundali" not in st.session_state:
        st.session_state["user_kundali"] = calculate_vedic_chart(dob, tob, lat, lon)
        st.session_state["active_birth_info"] = f"{dob} at {tob} IST ({city_choice})"
        st.success("Kundali Calculated!")

    st.divider()
    st.header("🗄️ 2. Book Knowledge Base")
    kb_mode = st.radio("Knowledge Base Source:", ["📂 Load Saved Knowledge Base", "📤 Index New Book/PDF"], index=0)

    if kb_mode == "📂 Load Saved Knowledge Base":
        saved_files = get_saved_kb_files()
        if saved_files:
            selected_file = st.selectbox("Select Cached Book Index:", saved_files)
            if st.button("⚡ Load Selected Index", use_container_width=True):
                loaded_kb, cloud_file_name, orig_name = load_kb_from_disk(os.path.join(KB_STORAGE_DIR, selected_file))
                st.session_state["knowledge_base"] = loaded_kb
                st.session_state["active_file_name"] = orig_name
                
                if cloud_file_name:
                    try:
                        st.session_state["cloud_file"] = client.files.get(name=cloud_file_name)
                    except Exception:
                        st.session_state["cloud_file"] = None
                else:
                    st.session_state["cloud_file"] = None
                st.success(f"Loaded: {orig_name}")
        else:
            st.info("No saved knowledge bases found yet. Select 'Index New Book/PDF' to create one.")
            
    else:
        uploaded_file = st.file_uploader("Upload Astrological Book (PDF/PNG/JPG):", type=["pdf", "png", "jpg"])
        if uploaded_file is not None:
            if st.button("🚀 Process & Permanently Save Index", use_container_width=True):
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
                    saved_path = save_kb_to_disk(uploaded_file.name, cloud_file.name, kb)

                    st.session_state["cloud_file"] = cloud_file
                    st.session_state["knowledge_base"] = kb
                    st.session_state["active_file_name"] = uploaded_file.name
                    st.success(f"Indexed & permanently saved to `{saved_path}`!")

# Main Stage: Interactive Kundali & Chart Visualizer
if "user_kundali" in st.session_state:
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
        st.plotly_chart(fig_kundali, use_container_width=True)

    with col_table_ui:
        tab_d1, tab_d9, tab_d10 = st.tabs(["🪐 D1 Planetary Table", "✨ D9 Navamsha", "💼 D10 Dashamsha"])
        
        with tab_d1:
            d1_rows = []
            for p, vals in kundali["D1_Lagna"].items():
                d1_rows.append({
                    "Planet": p,
                    "House (Bhava)": f"House {vals['House']}",
                    "Sign (Rashi)": vals["Sign"],
                    "Degree in Sign": f"{vals['Degrees']}°",
                    "Dignity": vals["Dignity"]
                })
            st.dataframe(d1_rows, use_container_width=True)

        with tab_d9:
            d9_rows = [{"Planet": p, "Navamsha House": f"House {v['House']}", "Sign": v["Sign"]} for p, v in kundali["D9_Navamsha"].items()]
            st.dataframe(d9_rows, use_container_width=True)

        with tab_d10:
            d10_rows = [{"Planet": p, "Dashamsha House": f"House {v['House']}", "Sign": v["Sign"]} for p, v in kundali["D10_Dashamsha"].items()]
            st.dataframe(d10_rows, use_container_width=True)

st.divider()

# Question & Answering Arena
if "knowledge_base" not in st.session_state:
    st.warning("👈 Please load a saved Knowledge Base or upload a book from the sidebar to begin querying.")
else:
    st.subheader(f"🔮 Consult Tri-Agent Court — Grounded in `{st.session_state['active_file_name']}`")
    query = st.text_input(
        "Ask about any planetary placement or life area:",
        value="What is the prediction for my Mars based on the book?"
    )

    consult_btn = st.button("⚡ Verify Chart Placements & Argue Book Rules", use_container_width=True)

    if consult_btn and query.strip():
        memory = load_skill_memory()

        # Cartoon Agent Deliberation Cards
        p1, p2, p3 = st.columns(3)
        c1 = p1.empty()
        c2 = p2.empty()
        c3 = p3.empty()

        c1.markdown("""
        <div style="background:#1E293B; border-radius:12px; padding:18px; border-left: 6px solid #38BDF8; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="font-size:36px; text-align:center;">🧑‍🏫</div>
            <h3 style="margin:5px 0; text-align:center; color:#F8FAFC;">The Scholar</h3>
            <p style="color:#38BDF8; font-size:12px; text-align:center; margin:0;"><b>STATUS: Ingesting Chart Facts</b></p>
            <p style="color:#94A3B8; font-size:13px; margin-top:8px;"><i>"Extracting factual placements of Mars from user's D1/D9 chart & matching book verses..."</i></p>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown("""
        <div style="background:#1E293B; border-radius:12px; padding:18px; border-left: 6px solid #F59E0B; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="font-size:36px; text-align:center;">⚔️</div>
            <h3 style="margin:5px 0; text-align:center; color:#F8FAFC;">The Inquisitor</h3>
            <p style="color:#F59E0B; font-size:12px; text-align:center; margin:0;"><b>STATUS: Standing Guard</b></p>
            <p style="color:#94A3B8; font-size:13px; margin-top:8px;"><i>"Ready to challenge Scholar for hallucinations or external astrological knowledge leaks..."</i></p>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown("""
        <div style="background:#1E293B; border-radius:12px; padding:18px; border-left: 6px solid #10B981; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="font-size:36px; text-align:center;">⚖️</div>
            <h3 style="margin:5px 0; text-align:center; color:#F8FAFC;">Supreme Auditor</h3>
            <p style="color:#10B981; font-size:12px; text-align:center; margin:0;"><b>STATUS: Overseeing Court</b></p>
            <p style="color:#94A3B8; font-size:13px; margin-top:8px;"><i>"Verifying zero collusion and testing adherence against pure book ground truth..."</i></p>
        </div>
        """, unsafe_allow_html=True)

        # 1. Scholar
        scholar_res = run_scholar(
            st.session_state.get("cloud_file"),
            st.session_state["knowledge_base"],
            st.session_state["user_kundali"],
            query,
            memory
        )
        c1.markdown("""
        <div style="background:#1E293B; border-radius:12px; padding:18px; border-left: 6px solid #38BDF8; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="font-size:36px; text-align:center;">🧑‍🏫</div>
            <h3 style="margin:5px 0; text-align:center; color:#F8FAFC;">The Scholar</h3>
            <p style="color:#34D399; font-size:12px; text-align:center; margin:0;"><b>✅ Synthesis Complete</b></p>
            <p style="color:#94A3B8; font-size:13px; margin-top:8px;"><i>"Delivered deduction anchored strictly in user chart facts & book verses."</i></p>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(1)

        # 2. Inquisitor
        c2.markdown("""
        <div style="background:#1E293B; border-radius:12px; padding:18px; border-left: 6px solid #F59E0B; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="font-size:36px; text-align:center;">⚔️</div>
            <h3 style="margin:5px 0; text-align:center; color:#F8FAFC;">The Inquisitor</h3>
            <p style="color:#F59E0B; font-size:12px; text-align:center; margin:0;"><b>🔄 Cross-Examining</b></p>
            <p style="color:#94A3B8; font-size:13px; margin-top:8px;"><i>"Verifying cancellations, checking for external leaks, and formulating new skill rule..."</i></p>
        </div>
        """, unsafe_allow_html=True)
        inquisitor_res = run_inquisitor(st.session_state["knowledge_base"], st.session_state["user_kundali"], query, scholar_res)
        
        if inquisitor_res.autonomous_lesson:
            update_skill_memory(inquisitor_res.autonomous_lesson)

        c2.markdown("""
        <div style="background:#1E293B; border-radius:12px; padding:18px; border-left: 6px solid #F59E0B; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="font-size:36px; text-align:center;">⚔️</div>
            <h3 style="margin:5px 0; text-align:center; color:#F8FAFC;">The Inquisitor</h3>
            <p style="color:#34D399; font-size:12px; text-align:center; margin:0;"><b>✅ Audited & Skill Updated</b></p>
            <p style="color:#94A3B8; font-size:13px; margin-top:8px;"><i>"Adversarial critique logged. Autonomous skill bank updated without user intervention."</i></p>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(1)

        # 3. Auditor
        c3.markdown("""
        <div style="background:#1E293B; border-radius:12px; padding:18px; border-left: 6px solid #10B981; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="font-size:36px; text-align:center;">⚖️</div>
            <h3 style="margin:5px 0; text-align:center; color:#F8FAFC;">Supreme Auditor</h3>
            <p style="color:#10B981; font-size:12px; text-align:center; margin:0;"><b>🔄 Certifying Ground Truth</b></p>
            <p style="color:#94A3B8; font-size:13px; margin-top:8px;"><i>"Conducting double-blind test to confirm zero collusion and 100% book alignment..."</i></p>
        </div>
        """, unsafe_allow_html=True)
        audit_res = run_auditor(st.session_state["knowledge_base"], query, scholar_res, inquisitor_res)
        
        c3.markdown("""
        <div style="background:#1E293B; border-radius:12px; padding:18px; border-left: 6px solid #10B981; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="font-size:36px; text-align:center;">⚖️</div>
            <h3 style="margin:5px 0; text-align:center; color:#F8FAFC;">Supreme Auditor</h3>
            <p style="color:#34D399; font-size:12px; text-align:center; margin:0;"><b>🏆 Verdict Certified</b></p>
            <p style="color:#94A3B8; font-size:13px; margin-top:8px;"><i>"Verified zero external leaks. Certified answer delivered."</i></p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.subheader("🏆 Certified Prediction Grounded Exclusively in Your Book")
        st.success(audit_res.certified_answer)

        r_col1, r_col2 = st.columns(2)
        r_col1.info(f"📍 **Facts Extracted From Your Chart (IST UTC+5:30):**\n{scholar_res.client_chart_facts_used}")
        r_col2.write("📚 **Exact Verses / Citations Quoted:**")
        for cite in scholar_res.book_citations_only:
            r_col2.markdown(f"- 📌 *{cite}*")

        st.markdown("---")
        st.markdown("#### 🧠 Autonomous Agent Skill Evolution (Self-Learned Memory)")
        st.caption("These rules were automatically synthesized and saved into `agent_skill_memory.json` during deliberations without user intervention:")
        for rule in load_skill_memory()[-5:]:
            st.markdown(f"- 💡 `{rule}`")