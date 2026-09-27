import json
import os
import time
from typing import List
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Automatically reads GEMINI_API_KEY from environment
client = genai.Client()

# Free-tier model with native multimodal vision & long document processing
FREE_MODEL = "gemini-2.5-flash"

# ============================================================================
# 1. Pydantic Schemas Enforcing Auditability and Preventing Collusion
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
# 2. Agent 1: The Scholar (Deep Visual & Text Reasoner)
# ============================================================================
def run_scholar(
    book_file_ref: types.File, user_query: str
) -> ScholarReasoning:
  system_prompt = """
    You are Agent 1: 'The Scholar'. You analyze ancient texts, manuscripts, and scanned books.
    
    RULES:
    1. Base all reasoning strictly on the provided scanned document.
    2. If astrological Kundalis, square/diamond charts, or ephemerides appear, extract every planetary 
       placement, house lordship, and aspect (drishti) visually.
    3. Synthesize rules across chapters (e.g., link foundational rules in Chapter 2 with yoga results in Chapter 14).
    4. Provide direct textual quotes and house-by-house chart readings.
    """

  response = client.models.generate_content(
      model=FREE_MODEL,
      contents=[
          book_file_ref,
          f"Analyze the source book and answer this query: {user_query}",
      ],
      config=types.GenerateContentConfig(
          system_instruction=system_prompt,
          response_mime_type="application/json",
          response_schema=ScholarReasoning,
          temperature=0.1,
      ),
  )
  return ScholarReasoning.model_validate_json(response.text)


# ============================================================================
# 3. Agent 2: The Inquisitor (Adversarial Adversary & Fact-Checker)
# ============================================================================
def run_inquisitor(
    book_file_ref: types.File, user_query: str, scholar_output: ScholarReasoning
) -> InquisitorEvaluation:
  system_prompt = """
    You are Agent 2: 'The Inquisitor'. Your sole job is to aggressively challenge Agent 1's work.
    
    ANTI-COLLUSION RULES:
    1. Treat Agent 1's claims with skepticism.
    2. Inspect the scanned chart/tables: Did Agent 1 misidentify an exalted planet, miss a debilitation cancellation 
       (Neechbhanga), or hallucinate a house placement?
    3. Cross-reference chapters: Did Agent 1 apply a general rule while ignoring a specific exception mentioned elsewhere?
    4. List every discrepancy clearly.
    """

  eval_prompt = f"""
    USER QUESTION: {user_query}
    
    AGENT 1'S WORK:
    - Textual Citations: {scholar_output.textual_citations}
    - Visual Chart Observations: {scholar_output.visual_chart_observations}
    - Cross-Chapter Synthesis: {scholar_output.cross_chapter_synthesis}
    - Final Verdict: {scholar_output.final_verdict}
    
    Verify this directly against the attached pages. Find and list all flaws.
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


# ============================================================================
# 4. Agent 3: The Supreme Auditor (Double-Blind Court)
# ============================================================================
def run_auditor(
    book_file_ref: types.File,
    user_query: str,
    scholar: ScholarReasoning,
    inquisitor: InquisitorEvaluation,
) -> AuditVerdict:
  system_prompt = """
    You are Agent 3: 'The Supreme Auditor'. You report only to the user.
    
    MANDATE:
    1. Detect collusion: Check whether Agent 2 gave a pass to incorrect claims or failed to inspect the chart.
    2. Check the raw scanned book independently to determine who is correct.
    3. Correct any errors and issue the definitive certified answer.
    """

  audit_payload = f"""
    USER QUESTION: {user_query}
    
    AGENT 1 REPORT:
    {scholar.model_dump_json(indent=2)}
    
    AGENT 2 EVALUATION:
    {inquisitor.model_dump_json(indent=2)}
    
    Independently inspect the document, assess both agents, check for collusion, and deliver the final verified answer.
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
# 5. Main Execution Loop with Free-Tier Rate-Limit Protection
# ============================================================================
def analyze_book_with_agents(file_path: str, question: str):
  print(f"[1/4] Uploading scanned document ({file_path}) to Free File API...")
  uploaded_file = client.files.upload(file=file_path)

  # Wait for file processing if it's a large multi-page PDF
  while uploaded_file.state.name == "PROCESSING":
    time.sleep(2)
    uploaded_file = client.files.get(name=uploaded_file.name)

  try:
    print(
        "[2/4] Agent 1 (Scholar) synthesizing cross-chapter rules and"
        " charts..."
    )
    scholar_res = run_scholar(uploaded_file, question)

    # Brief 2-second pause to easily stay within free tier 15 RPM
    time.sleep(2)

    print(
        "[3/4] Agent 2 (Inquisitor) stress-testing claims & auditing"
        " diagrams..."
    )
    inquisitor_res = run_inquisitor(uploaded_file, question, scholar_res)

    time.sleep(2)

    print(
        "[4/4] Agent 3 (Auditor) running anti-collusion review & certifying"
        " answer..."
    )
    audit_res = run_auditor(
        uploaded_file, question, scholar_res, inquisitor_res
    )

    return {
        "scholar": scholar_res,
        "inquisitor": inquisitor_res,
        "audit": audit_res,
    }

  finally:
    # Always delete file from cloud storage when done
    client.files.delete(name=uploaded_file.name)


# ============================================================================
# Test Runner
# ============================================================================
if __name__ == "__main__":
  # Place any scanned PDF or image with charts/tables here
  DOC_PATH = "sample_astrology_scan.pdf"
  PROMPT = (
      "Examine the chart on page 12 and synthesize rules from Chapter 4 (Bhavas)"
      " and Chapter 9 (Yogas): What is the status of the 10th lord, does any"
      " Neechbhanga yoga occur, and what is the outcome?"
  )

  if os.path.exists(DOC_PATH):
    results = analyze_book_with_agents(DOC_PATH, PROMPT)

    print("\n" + "=" * 65)
    print("                 FINAL AUDITED ANSWER")
    print("=" * 65)
    print(f"Certified Response:\n{results['audit'].certified_answer}\n")
    print(
        f"Integrity Score: {results['audit'].system_integrity_score}/100"
    )
    print(
        f"Collusion Flagged: {results['audit'].collusion_detected}"
    )
    print(
        f"Source Adherence: {results['audit'].ground_truth_alignment}"
    )

    print("\n" + "=" * 65)
    print("             AGENT 2 (INQUISITOR) CRITIQUE")
    print("=" * 65)
    print(f"Passed Inquisition: {results['inquisitor'].passed_inquisition}")
    print(
        "Chart Reading Accuracy:"
        f" {results['inquisitor'].chart_reading_accuracy_pct}%"
    )
    print(
        f"Contradictions Caught: {results['inquisitor'].contradictions_found}"
    )
    print(f"Critique: {results['inquisitor'].adversarial_critique}")
  else:
    print(f"To run, place a scanned PDF or image named '{DOC_PATH}' in this directory.")