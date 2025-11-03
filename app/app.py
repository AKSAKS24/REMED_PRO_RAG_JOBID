from __future__ import annotations

import os
import json
import datetime
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, status
from pydantic import BaseModel, Field

# LangChain + OpenAI + Chroma
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# =========================
# Config (env)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # e.g., "gpt-5"
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_rules")  # persisted vector store

# Optional LangSmith tracing (if you use it)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# =========================
# FastAPI app
# =========================

app = FastAPI(
    title="ABAP Remediator (LLM + Chroma RAG, PwC tagging)",
    version="2.2"
)

# =========================
# Models
# =========================

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    code: str
    llm_prompt: List[str] = Field(default_factory=list)

class RebuildResponse(BaseModel):
    ok: bool
    docs_indexed: int
    persist_directory: str

# =========================
# LLM prompt (strict JSON)
# =========================

SYSTEM_MSG = (
    "You are a precise ABAP remediation engine. "
    "Use the retrieved rules verbatim. Follow the bullets in 'llm_prompt' exactly. "
    "Return STRICT JSON only."
)

USER_TEMPLATE = """
<retrieved_rules>
{rules}
</retrieved_rules>

Remediate the ABAP code EXACTLY following the bullet points in 'llm_prompt'.
Rules you MUST follow:
- Replace legacy/wrong code with corrected ABAP per the rules and bullets.
- Output the FULL remediated code (not a diff).
- Strictly Never end any block explicitly like ( ENDMETHOD, ENDFORM...) Unless present in the code.
- Every ADDED or MODIFIED line must include an inline ABAP comment at the end of that line:  " Added By Pwc{today_date}
  (Use a single double-quote ABAP comment delimiter.)
- Keep behavior the same unless the bullets say otherwise.
- Use ECC-safe syntax unless the bullets allow otherwise.
- Always Follow Rule 1 for All Select Single statement to convert it Select Endselect.
- Always follow Rule 5 for any offset (eg: +2(3)), use Substring Function.
- Always follow Rule 6 for any CONCATENATE used.
- Return ONLY strict JSON with keys:
{{
  "remediated_code": "<full updated ABAP code with PwC comments on added/modified lines>"
}}

Context:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}
- Today's date (PwC tag): {today_date}

Original ABAP code:
{code}

llm_prompt (bullets):
{llm_prompt_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

# =========================
# Background job store
# =========================

_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}
# job schema:
# {
#   "status": "pending" | "running" | "done" | "failed",
#   "attempts": int,
#   "result": Optional[Dict[str, Any]],  # final output JSON (original fields + rem_code)
#   "error": Optional[str]
# }

# =========================
# LLM runner
# =========================

def today_iso() -> str:
    return datetime.date.today().isoformat()

def _extract_json_str(s: str) -> str:
    """
    Best-effort extractor: some LLMs may wrap JSON in code fences.
    """
    t = s.strip()
    if t.startswith("```"):
        t = t.split("```", 2)
        if len(t) == 3:
            t = t[1] if not t[1].lstrip().startswith("{") else t[1]
            t = "\n".join(
                line for line in t.splitlines()
                if not line.strip().lower().startswith("json")
            ).strip()
        else:
            t = s
    return t.strip()

def remediate_with_rag(unit: Unit, request_timeout: int = 600) -> str:
    if not unit.llm_prompt:
        raise HTTPException(status_code=400, detail="llm_prompt must be a non-empty list of instructions.")

    # Retrieve rules
    ruleset_loader = TextLoader("ruleset.txt", encoding="utf-8")
    documents = ruleset_loader.load()

    # Split Rules into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0
    )
    docs = text_splitter.split_documents(documents)
    rules_text = "\n\n".join([doc.page_content for doc in docs])

    # request_timeout enforces the 600s limit per attempt; max_retries=0 so we control retries
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=1, request_timeout=request_timeout, max_retries=0)
    payload = {
        "rules": rules_text or "No rules retrieved.",
        "pgm_name": unit.pgm_name,
        "inc_name": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name or "",
        "code": unit.code or "",
        "today_date": today_iso(),
        "llm_prompt_json": json.dumps(unit.llm_prompt, ensure_ascii=False, indent=2),
    }
    msgs = prompt.format_messages(**payload)
    resp = llm.invoke(msgs)

    content = resp.content or ""
    content = _extract_json_str(content)
    try:
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model did not return valid JSON: {e}")

    rem = data.get("remediated_code", "")
    if not isinstance(rem, str) or not rem.strip():
        raise HTTPException(status_code=502, detail="Model returned empty or invalid 'remediated_code'.")
    return rem

# =========================
# Background job worker
# =========================

def _finalize_job_success(job_id: str, unit: Unit, rem_code: str) -> None:
    result = unit.model_dump()
    result["rem_code"] = rem_code
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = result
        _JOBS[job_id]["error"] = None

def _finalize_job_failure(job_id: str, unit: Unit, error_msg: str) -> None:
    result = unit.model_dump()
    result["rem_code"] = ""
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = result
        _JOBS[job_id]["error"] = error_msg

def _process_job(job_id: str, unit: Unit, timeout_sec: int = 600, max_attempts: int = 3) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["attempts"] = 0

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        with _JOBS_LOCK:
            _JOBS[job_id]["attempts"] = attempt
        try:
            rem_code = remediate_with_rag(unit, request_timeout=timeout_sec)
            _finalize_job_success(job_id, unit, rem_code)
            return
        except Exception as e:
            last_error = str(e)
            continue

    _finalize_job_failure(job_id, unit, last_error or "Max attempts exceeded")

# =========================
# Endpoints
# =========================

@app.post("/remediate_direct")
def remediate(unit: Unit) -> Dict[str, Any]:
    """
    Synchronous remediation.
    Input JSON:
      {
        "pgm_name": "...",
        "inc_name": "...",
        "type": "...",
        "name": "",
        "class_implementation": "",
        "code": "<ABAP code>",
        "llm_prompt": [ "...bullet...", "...bullet..." ]
      }

    Output JSON:
      original fields + "rem_code": "<full remediated ABAP>"
    """
    rem_code = remediate_with_rag(unit)
    obj = unit.model_dump()
    obj["rem_code"] = rem_code
    return obj

@app.post("/remediate")
def remediate_job(unit: Unit, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Submit a background remediation job.
    Input JSON is exactly the same as /remediate.
    Returns:
      { "job_id": "<uuid>" }
    """
    job_id = uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "pending",
            "attempts": 0,
            "result": None,
            "error": None,
        }
    background_tasks.add_task(_process_job, job_id, unit, 600, 3)
    return {"job_id": job_id}

@app.get("/remediate/{job_id}")
def get_remediate_job(job_id: str, response: Response) -> Dict[str, Any]:
    """
    Retrieve the result of a background remediation job.
    On completion, returns the exact same output JSON as /remediate:
      original fields + "rem_code".
    If not yet complete, returns 202 with:
      { "status": "...", "attempts": N }
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    status_str = job["status"]
    if status_str == "done" and job.get("result") is not None:
        return job["result"]

    # Not yet complete
    response.status_code = status.HTTP_202_ACCEPTED
    return {
        "status": status_str,
        "attempts": job.get("attempts", 0),
    }