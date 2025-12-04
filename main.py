from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import json
import os

# ------------ Data Loading ------------

FILES = [
    "iims_clean_full.json",
    "iits_clean_full.json",
    "privates_clean_full.json",
    "remaining1_clean_full.json",
    "remaining2_clean_full.json"
]

def load_ds():
    data = []
    for fname in FILES:
        path = os.path.join("data", fname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data.extend(json.load(f))
            except Exception:
                continue
    return data

def expand(ds):
    out = []
    for r in ds:
        specs = r.get("specializations") or []
        if specs:
            for s in specs:
                e = dict(r)
                e["subprogram"] = s
                e["entity_id"] = f"{r.get('entity_id', r.get('brand',''))}__{s.replace(' ','_')}"
                out.append(e)
        else:
            if "entity_id" not in r:
                r["entity_id"] = r.get("brand","")
            out.append(r)
    return out

DATA = expand(load_ds())

# ------------ Engine Logic ------------

def is_govt(e): 
    return bool(e.get("is_govt", False))

def best_exam(e, inp):
    acc = e.get("accepted_exams") or []
    best = None
    used = None
    for ex in acc:
        pct = (inp.exam_percentiles or {}).get(ex)
        if pct is None:
            continue
        if best is None or pct > best:
            best, used = pct, ex
    return best, used

def relax(e, inp):
    r = {"cat":0,"gen":0,"acad":0,"work":0}
    if is_govt(e) and e.get("has_reservation", False):
        cat = inp.category or "Gen"
        r["cat"] = {
            "EWS":5,"NC-OBC":10,"SC":25,"ST":35,"PWD":40
        }.get(cat,0)
    if e.get("gender_relax", False) and (inp.gender or "").lower().startswith("f"):
        r["gen"] = e.get("gender_relax_value", 2)
    if e.get("acad_relax", False) and inp.academic_stream != "Engineer":
        r["acad"] = e.get("acad_relax_value", 3)
    if e.get("workex_relax", False):
        yrs = inp.workex_years or 0
        r["work"] = min(4, int(yrs))
    return r

def eligibility_ok(e, pct, inp):
    elig = e.get("eligibility_cutoffs",{}) or {}
    secs = elig.get("sectionals",{})
    cand = inp.section_percentiles or {}
    for k,v in secs.items():
        if cand.get(k,0) < v:
            return False
    if pct < elig.get("overall", 0):
        return False
    return True

def roi(e):
    fees = e.get("total_fees_lac")
    ctc = e.get("avg_ctc_lac")
    if not fees or not ctc or fees <= 0:
        return None, "Unknown"
    r = ctc / fees
    if r >= 2:
        return r, "High ROI"
    if r >= 1.2:
        return r, "Medium ROI"
    return r, "Low ROI"

def weight_fit(e, inp):
    wg = e.get("weight_grid") or {}
    if not wg:
        return 0, "Unknown"
    tot = sum(wg.values()) or 1
    score = 0.0

    # exam ~ neutral
    score += 0.6 * wg.get("exam", 0)

    # academics
    a10 = inp.acad_10 or 0
    a12 = inp.acad_12 or 0
    ag  = inp.acad_grad or 0
    lvl = (a10 >= 80) + (a12 >= 75) + (ag >= 70)
    score += (0.3 + 0.2 * lvl) * wg.get("acads", 0) / 3.0

    # workex
    yrs = inp.workex_years or 0
    wx = 0.3 if yrs == 0 else (0.7 if yrs <= 3 else 0.5)
    score += wx * wg.get("workex", 0)

    fit = score / tot
    if fit >= 0.75: 
        return fit, "Strong fit"
    if fit >= 0.5:
        return fit, "Moderate fit"
    return fit, "Weak fit"

def spec_bonus(e, inp):
    pref = (inp.specialization_preference or "").lower()
    if not pref:
        return 0, "No preference"
    sf = e.get("strength_flags") or {}
    st = sf.get(pref, 0)
    if st >= 3: return 5, "Strong match"
    if st == 2: return 3, "Moderate match"
    if st == 1: return 1, "Weak match"
    return 0, "No match"

def base_prob(margin):
    if margin >= 5:  return 90
    if margin >= 0:  return 65
    if margin >= -3: return 45
    if margin >= -8: return 25
    return 5

def tier_from_prob(p):
    if p >= 70: return "SAFE"
    if p >= 40: return "TARGET"
    if p >= 15: return "REACH"
    if p >= 1:  return "UNLIKELY"
    return "REJECT"

def compute_program(e, inp):
    pct, exam_used = best_exam(e, inp)
    if pct is None:
        return None  # no valid exam

    if not eligibility_ok(e, pct, inp):
        return {
            "prob": 0,
            "tier": "REJECT",
            "reason": "Failed eligibility",
            "exam_used": exam_used,
            "roi_cat": "Unknown",
            "spec": "No preference",
            "fit": "Unknown"
        }

    call = e.get("actual_call_percentile")
    if call is None:
        # minimal info
        prob = 15
        return {
            "prob": prob,
            "tier": tier_from_prob(prob),
            "reason": "Unknown cutoff; conservative low score",
            "exam_used": exam_used,
            "roi_cat": "Unknown",
            "spec": "No preference",
            "fit": "Unknown"
        }

    r = relax(e, inp)
    required = call - r["cat"] - r["gen"] - r["acad"] - r["work"] + e.get("cons_margin", 3)
    margin = pct - required
    base = base_prob(margin)

    roi_ratio, roi_cat = roi(e)
    fit_score, fit_txt = weight_fit(e, inp)
    bonus_spec, spec_txt = spec_bonus(e, inp)

    bonus_roi = 3 if roi_cat == "High ROI" else (1 if roi_cat == "Medium ROI" else 0)
    bonus_fit = 2 if fit_txt == "Strong fit" else (1 if fit_txt == "Moderate fit" else 0)

    prob = max(0, min(100, base + bonus_spec + bonus_roi + bonus_fit))

    return {
        "prob": prob,
        "tier": tier_from_prob(prob),
        "reason": f"Margin {margin:+.1f} vs required {required:.1f}",
        "exam_used": exam_used,
        "roi_cat": roi_cat,
        "spec": spec_txt,
        "fit": fit_txt
    }

# ------------ Pydantic Models & API ------------

class PredictInput(BaseModel):
    exam_percentiles: Dict[str, float] = Field(default_factory=dict)
    category: str
    gender: Optional[str] = None
    academic_stream: Optional[str] = None
    workex_years: Optional[float] = 0
    section_percentiles: Optional[Dict[str, float]] = Field(default_factory=dict)
    specialization_preference: Optional[str] = None
    max_fees_lac: Optional[float] = None
    acad_10: Optional[float] = None
    acad_12: Optional[float] = None
    acad_grad: Optional[float] = None

class ProgramOut(BaseModel):
    college: Optional[str]
    campus: Optional[str]
    program: Optional[str]
    probability_percent: float
    tier: str
    reason: str
    roi_category: str
    specialisation_alignment: str
    weight_fit_summary: str
    exam_used: Optional[str]
    deadline_status: str = "UNKNOWN"
    web_updates: str = ""

app = FastAPI(title="MBA Recommendation Engine")

@app.post("/predict", response_model=List[ProgramOut])
def predict(inp: PredictInput):
    results: List[ProgramOut] = []
    for e in DATA:
        # fees filter
        if inp.max_fees_lac and e.get("total_fees_lac") and e["total_fees_lac"] > inp.max_fees_lac:
            continue

        res = compute_program(e, inp)
        if res is None:
            continue

        out = ProgramOut(
            college = e.get("brand"),
            campus  = e.get("location"),
            program = e.get("program"),
            probability_percent = res["prob"],
            tier = res["tier"],
            reason = res["reason"],
            roi_category = res["roi_cat"],
            specialisation_alignment = res["spec"],
            weight_fit_summary = res["fit"],
            exam_used = res["exam_used"],
            deadline_status = "UNKNOWN",
            web_updates = ""
        )
        results.append(out)

    # sort desc by prob
    results.sort(key=lambda r: r.probability_percent, reverse=True)
    return results[:20]
