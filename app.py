"""
HiringAgent - AI Resume Screening Platform
Production-grade Flask application v3.0
"""

import os
import re
import json
import uuid
import logging
import threading
import secrets
import time
import sys
import requests
from functools import wraps
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from flask import (
    Flask, render_template, request, jsonify, 
    send_file, session, redirect, url_for, make_response
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import EvaluationData
from evaluator import ResumeEvaluator
from llm_utils import initialize_llm_provider, extract_json_from_response
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from transform import convert_json_resume_to_text, convert_github_data_to_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("app.log", encoding="utf-8")]
)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    UPLOAD_FOLDER = "uploads"; HISTORY_FOLDER = "evaluations"
    for folder in [UPLOAD_FOLDER, HISTORY_FOLDER, "logs", "cache"]:
        Path(folder).mkdir(exist_ok=True)
    app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, HISTORY_FOLDER=HISTORY_FOLDER, MAX_CONTENT_LENGTH=20*1024*1024, SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax', USERS_FILE="users.json", APPLICATIONS_FILE="applications.json")
    SECRET_KEY_FILE = ".flask_secret_key"
    if os.path.exists(SECRET_KEY_FILE):
        with open(SECRET_KEY_FILE) as f: app.secret_key = f.read().strip()
    else:
        app.secret_key = secrets.token_hex(32)
        with open(SECRET_KEY_FILE, "w") as f: f.write(app.secret_key)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    return app

app = create_app()

CATEGORY_MAX = {"open_source":35,"self_projects":30,"production":25,"technical_skills":10}
STOPWORDS = {"the","and","for","with","that","this","from","have","will","you","your","are","our","a","an","to","of","in","on","as","be","or","is","we","at","by","it","its","who","role","team","work","years","year","experience","job","using","able","strong","good","must","should","can","into","such","other","including","responsible","ability","knowledge","looking","candidate"}
TECH_CATEGORIES = {"Languages":{"python","java","javascript","typescript","c","c++","c#","go","golang","rust","ruby","php","kotlin","swift","scala","r","sql","html","css","bash","shell","matlab","perl","dart"},"Frameworks & Libraries":{"react","reactjs","angular","vue","vuejs","django","flask","fastapi","spring","springboot","express","expressjs","nextjs","nodejs","node","rails","laravel","tensorflow","pytorch","keras","pandas","numpy","scikit-learn","sklearn","bootstrap","tailwind","jquery","redux","graphql"},"Tools & Platforms":{"docker","kubernetes","k8s","aws","azure","gcp","git","github","gitlab","jenkins","terraform","ansible","linux","nginx","postgresql","postgres","mysql","mongodb","redis","elasticsearch","kafka","rabbitmq","jira","figma","webpack","ci/cd","cicd","firebase","supabase","vercel","heroku"},"Concepts":{"microservices","rest","restful","api","apis","agile","scrum","devops","oop","testing","unittest","tdd","ml","ai","machine","learning","nlp","concurrency","multithreading","distributed","systems","security","authentication","oauth","websockets","grpc","algorithms","datastructures"}}
ACTIVE_JOBS_API_KEY = os.getenv("ACTIVE_JOBS_API_KEY","")
ACTIVE_JOBS_HOST = "active-jobs-db.p.rapidapi.com"
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID",""); ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY","")
EXPERIENCE_FILTER_MAP = {"entry":"0-2","mid":"2-5","senior":"5-10","executive":"10+"}
WORK_ARRANGEMENT_MAP = {"remote":"Remote","hybrid":"Hybrid","onsite":"On-site"}
SUPPORTED_COUNTRIES = {"worldwide":"Worldwide","us":"United States","uk":"United Kingdom","in":"India","ca":"Canada","de":"Germany","sg":"Singapore","au":"Australia","remote_only":"Remote Only"}
JOB_TYPES = {"full_time":"Full-time","part_time":"Part-time","contract":"Contract","internship":"Internship","new_grad":"New Grad"}
DATE_POSTED_FILTERS = {"today":1,"last_3_days":3,"last_7_days":7,"last_14_days":14,"last_30_days":30}
JOBS = {}; JOBS_LOCK = threading.Lock()
JOB_SEARCH_CACHE = {}; JOB_SEARCH_CACHE_TTL = 900

# ============================================================
# AUTH
# ============================================================
def load_users():
    f = app.config.get('USERS_FILE','users.json')
    if not os.path.exists(f): return []
    try:
        with open(f,encoding="utf-8") as fp: return json.load(fp)
    except: return []

def save_users(u):
    with open(app.config.get('USERS_FILE','users.json'),"w",encoding="utf-8") as fp: json.dump(u,fp,indent=2,ensure_ascii=False)

def find_user(k):
    k = k.strip().lower()
    for u in load_users():
        if u["username"].lower()==k or u.get("email","").lower()==k: return u
    return None

def login_required_page(v):
    @wraps(v)
    def w(*a,**kw):
        if not session.get("user_id"): return redirect(url_for("login",next=request.path))
        return v(*a,**kw)
    return w

def login_required_api(v):
    @wraps(v)
    def w(*a,**kw):
        if not session.get("user_id"): return jsonify({"success":False,"error":"Not logged in"}),401
        return v(*a,**kw)
    return w

def current_user_id(): return session.get("user_id")

# ============================================================
# HELPERS
# ============================================================
def find_profile(profiles,network):
    if not profiles: return None
    return next((p for p in profiles if p.network and p.network.lower()==network.lower()),None)

def is_valid_resume_data(d):
    if not d: return False
    return any(getattr(d,s,None) is not None for s in ["basics","work","education","skills","projects"])

def run_evaluation(resume_data,github_data=None):
    mp = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    ev = ResumeEvaluator(model_name=DEFAULT_MODEL,model_params=mp)
    t = convert_json_resume_to_text(resume_data)
    if github_data: t += convert_github_data_to_text(github_data)
    return ev.evaluate_resume(t), t

def extract_keywords(text):
    return {w for w in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{1,}",text.lower()) if w not in STOPWORDS and len(w)>2}

def categorize_keywords(kw):
    g = {c:[] for c in TECH_CATEGORIES}; g["Other"]=[]
    for w in sorted(kw):
        placed=False
        for c,t in TECH_CATEGORIES.items():
            if w in t: g[c].append(w); placed=True; break
        if not placed: g["Other"].append(w)
    return {c:k for c,k in g.items() if k}

def match_job_description(resume_text,jd_text):
    rk = extract_keywords(resume_text); jk = extract_keywords(jd_text)
    if not jk: return {"match_score":0,"matched_keywords":[],"missing_keywords":[],"matched_grouped":{},"missing_grouped":{}}
    m = rk & jk; ms = jk - rk; sc = round(len(m)/len(jk)*100,1)
    mg = categorize_keywords(ms); mt = {k:v for k,v in mg.items() if k!="Other"}
    return {"match_score":sc,"matched_keywords":sorted(m)[:40],"missing_keywords":sorted(ms)[:40],"matched_grouped":categorize_keywords(m),"missing_grouped":mg,"suggestions":mt or mg}

def evaluation_to_dict(e):
    if e is None: return {}
    if hasattr(e,"model_dump"): return e.model_dump()
    if hasattr(e,"dict"): return e.dict()
    return json.loads(json.dumps(e,default=str))

def compute_total_score(ev):
    s = ev.get("scores") or {}; mt=0; t=0
    for k in CATEGORY_MAX:
        c = s.get(k)
        if not c: continue
        cm = c.get("max",CATEGORY_MAX[k]); mt+=cm; t+=min(c.get("score",0),cm)
    b = (ev.get("bonus_points") or {}).get("total",0) or 0
    d = (ev.get("deductions") or {}).get("total",0) or 0
    return {"total_score":round(max(0,t+b-d),1),"max_score":mt}

def _first_of(d,keys,default=None):
    for k in keys:
        v = d.get(k)
        if v: return v
    return default

# ============================================================
# ROBUST JSON PARSER - NEVER FAILS
# ============================================================
def robust_json_parse(text: str) -> dict:
    """Parse JSON from LLM response - multiple fallback strategies, never raises"""
    cleaned = text.strip()
    
    # Remove markdown fences
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?```\s*', '', cleaned)
    cleaned = cleaned.strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract between { and }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Fix common issues
    fixed = cleaned
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    fixed = fixed.replace("'", '"')
    s = fixed.find('{'); e = fixed.rfind('}')
    if s != -1 and e != -1: fixed = fixed[s:e+1]
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Fix truncated JSON
    lines = fixed.split('\n')
    while lines and not lines[-1].strip(): lines.pop()
    if lines:
        ll = lines[-1].strip()
        if ll.count('"') % 2 != 0:
            lines.pop()
            if lines and lines[-1].rstrip().endswith(','): lines[-1] = lines[-1].rstrip()[:-1]
    fixed = '\n'.join(lines)
    ob = fixed.count('{') - fixed.count('}')
    oa = fixed.count('[') - fixed.count(']')
    fixed = fixed.rstrip().rstrip(',')
    suffix = ']' * max(0, oa) + '}' * max(0, ob)
    try:
        return json.loads(fixed + suffix)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Manual regex extraction - NEVER FAILS
    logger.warning("All JSON parse strategies failed, using manual extraction")
    return _manual_extract(text)

def _manual_extract(text: str) -> dict:
    """Manual extraction using regex - guaranteed to return something"""
    result = {}
    
    # Extract numbers
    for key in ['match_score','score','overall_score','technical_accuracy','communication_clarity',
                'confidence_level','correctness_score','code_quality_score','efficiency_score',
                'edge_cases_score','filler_words_count','word_count','duration_minutes']:
        m = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', text)
        if m:
            try: result[key] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
            except: result[key] = m.group(1)
    
    # Extract strings
    for key in ['test_title','title','candidate_name','type','difficulty','language',
                'feedback','feedback_summary','explanation','suggested_solution',
                'suggested_answer_outline','subject_line','cover_letter','why_this_company',
                'duration_estimate','time_complexity','space_complexity','pace','structure',
                'overall_assessment','interview_id','job_title','company','job_id']:
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if m: result[key] = m.group(1)
    
    # Extract arrays
    for key in ['strengths','improvements','missing_skills','tips','areas_to_improve',
                'questions','cover_letters','messages','linkedin_search_queries',
                'recruiter_types','target_companies','networking_tips','hashtags_to_follow',
                'key_skills_highlighted','follow_up_tips','hints','test_cases',
                'expected_topics','good_answer_indicators','key_phrases_used',
                'improvement_suggestions','filler_words','coding_questions',
                'core_subject_questions','project_questions','behavioral_questions']:
        # Find array content
        arr_match = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if arr_match:
            items = re.findall(r'"([^"]*)"', arr_match.group(1))
            if items: result[key] = items
    
    # Extract evaluation_criteria
    ec_match = re.search(r'"evaluation_criteria"\s*:\s*\{([^}]+)\}', text)
    if ec_match:
        criteria = {}
        for m in re.finditer(r'"(\w+)"\s*:\s*(\d+)', ec_match.group(1)):
            criteria[m.group(1)] = int(m.group(2))
        if criteria: result['evaluation_criteria'] = criteria
    
    # Extract scoring_rubric
    sr_match = re.search(r'"scoring_rubric"\s*:\s*\{([^}]+)\}', text)
    if sr_match:
        rubric = {}
        for m in re.finditer(r'"(\w+)"\s*:\s*(\d+)', sr_match.group(1)):
            rubric[m.group(1)] = int(m.group(2))
        if rubric: result['scoring_rubric'] = rubric
    
    # Ensure minimum structure
    if 'questions' not in result and 'coding_questions' not in result:
        result['questions'] = [{"id":1,"question":"Tell me about your experience.","category":"general"}]
    
    if 'match_score' not in result:
        result['match_score'] = 50
    
    logger.info(f"Manual extraction result keys: {list(result.keys())}")
    return result

def generate_interview_questions(resume_text):
    p = initialize_llm_provider(DEFAULT_MODEL)
    mp = MODEL_PARAMETERS.get(DEFAULT_MODEL,{"temperature":0.4,"top_p":0.9})
    sm = "Technical interviewer. Respond ONLY with valid JSON, no markdown."
    up = f"""Generate interview questions. Return EXACTLY:
{{"coding_questions":["q1","q2","q3"],"core_subject_questions":["q1","q2","q3"],"project_questions":["q1","q2"],"behavioral_questions":["q1","q2"]}}
Resume: {resume_text[:3000]}"""
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"system","content":sm},{"role":"user","content":up}],"options":{"stream":False,"temperature":mp.get("temperature",0.4),"top_p":mp.get("top_p",0.9)}}
    r = p.chat(**cp)
    return robust_json_parse(extract_json_from_response(r["message"]["content"]))

# ============================================================
# JOB SEARCH
# ============================================================
def fetch_jobs_adzuna(role,country="us",page=1,rpp=50):
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY: return []
    try:
        u = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        r = requests.get(u,params={"app_id":ADZUNA_APP_ID,"app_key":ADZUNA_APP_KEY,"results_per_page":min(rpp,100),"what":role},timeout=15)
        r.raise_for_status()
        return [{"source":"adzuna","job_id":j.get("id",uuid.uuid4().hex[:12]),"title":j.get("title",""),"company":j.get("company",{}).get("display_name",""),"location":j.get("location",{}).get("display_name",""),"description":j.get("description",""),"apply_link":j.get("redirect_url",""),"posted_at":j.get("created",""),"salary":f"${j.get('salary_min')}-{j.get('salary_max')}" if j.get('salary_min') else "","is_remote":"remote" in j.get("location",{}).get("display_name","").lower(),"contract_type":j.get("contract_type","")} for j in r.json().get("results",[])]
    except: return []

def fetch_jobs_remoteok(role):
    try:
        r = requests.get("https://remoteok.com/api",params={"tag":role.lower()},headers={"User-Agent":"Mozilla/5.0"},timeout=15)
        r.raise_for_status(); d = r.json()
        return [{"source":"remoteok","job_id":str(j.get("id",uuid.uuid4().hex[:12])),"title":j.get("position",""),"company":j.get("company",""),"location":"Remote","description":j.get("description",""),"apply_link":j.get("url",""),"posted_at":j.get("date",""),"salary":j.get("salary",""),"is_remote":True,"contract_type":j.get("employment_type","")} for j in d[1:] if isinstance(j,dict)]
    except: return []

def fetch_jobs_arbeitnow(role,location=""):
    try:
        p = {}
        if role: p["search"]=role
        if location: p["location"]=location
        r = requests.get("https://www.arbeitnow.com/api/job-board-api",params=p,timeout=15)
        r.raise_for_status()
        return [{"source":"arbeitnow","job_id":j.get("slug",uuid.uuid4().hex[:12]),"title":j.get("title",""),"company":j.get("company_name",""),"location":f"{j.get('location','')}{' / Remote' if j.get('remote') else ''}","description":j.get("description",""),"apply_link":j.get("url",""),"posted_at":j.get("created_at",""),"salary":"","is_remote":j.get("remote",False),"contract_type":",".join(j.get("job_types",[]))} for j in r.json().get("data",[])]
    except: return []

def search_jobs_active_db(role,location="",country="worldwide",experience="",remote="",job_type="",date_posted="",page=1,limit=100):
    if not ACTIVE_JOBS_API_KEY: raise RuntimeError("ACTIVE_JOBS_API_KEY not set")
    ck = f"{role}|{location}|{country}|{experience}|{remote}|{job_type}|{date_posted}|{page}|{limit}"
    cc = JOB_SEARCH_CACHE.get(ck)
    if cc and (time.time()-cc[0])<JOB_SEARCH_CACHE_TTL: return cc[1]
    off = (page-1)*limit
    p = {"limit":str(min(limit,100)),"offset":str(off),"description_type":"text"}
    if role: p["title_filter"]=role
    if country!="worldwide" and country!="remote_only": p["location_filter"]=f"{location}, {SUPPORTED_COUNTRIES.get(country,country)}" if location else SUPPORTED_COUNTRIES.get(country,country)
    elif location: p["location_filter"]=location
    if experience in EXPERIENCE_FILTER_MAP: p["ai_experience_level_filter"]=EXPERIENCE_FILTER_MAP[experience]
    if remote in WORK_ARRANGEMENT_MAP: p["ai_work_arrangement_filter"]=WORK_ARRANGEMENT_MAP[remote]
    elif country=="remote_only": p["ai_work_arrangement_filter"]="Remote"
    h = {"X-RapidAPI-Key":ACTIVE_JOBS_API_KEY,"X-RapidAPI-Host":ACTIVE_JOBS_HOST}
    try:
        r = requests.get(f"https://{ACTIVE_JOBS_HOST}/active-ats-7d",headers=h,params=p,timeout=30)
        r.raise_for_status(); d = r.json()
        if isinstance(d,list): jd,tc = d,len(d)
        elif isinstance(d,dict): jd,tc = d.get("data",d.get("jobs",[])),d.get("total",d.get("count",len(d.get("data",d.get("jobs",[])))))
        else: jd,tc = [],0
        hm = tc>(off+limit) if tc>0 else len(jd)>=limit
        tp = (tc+limit-1)//limit if tc>0 else 1
        res = {"jobs":jd,"pagination":{"current_page":page,"total_pages":tp,"total_jobs":tc or len(jd),"has_next_page":hm,"has_previous_page":page>1,"limit":limit}}
        JOB_SEARCH_CACHE[ck] = (time.time(),res)
        return res
    except: return {"jobs":[],"pagination":{"current_page":page,"total_pages":0,"total_jobs":0,"has_next_page":False,"has_previous_page":False,"limit":limit}}

def normalize_active_db_job(job):
    locs = job.get("locations") or []
    ad = (locs[0].get("address") if locs else {}) or {}
    return {"source":"active_jobs_db","job_id":str(job.get("id","")),"title":job.get("title",""),"company":job.get("organization",""),"location":ad.get("addressLocality",""),"city":ad.get("addressLocality",""),"country":ad.get("addressCountry",""),"is_remote":bool(_first_of(job,["ai_work_arrangement","remote_derived"],"") and "remote" in str(_first_of(job,["ai_work_arrangement","remote_derived"],"")).lower()),"apply_link":_first_of(job,["url","apply_url","job_url"],job.get("organization_url","")),"posted_at":job.get("date_posted",""),"description":_first_of(job,["description","description_text"],""),"salary":"","contract_type":""}

def score_job_for_candidate(resume_text,job):
    p = initialize_llm_provider(DEFAULT_MODEL)
    mp = MODEL_PARAMETERS.get(DEFAULT_MODEL,{"temperature":0.2,"top_p":0.9})
    jt = f"Title: {job.get('title','')}\nCompany: {job.get('company','')}\nLocation: {job.get('location','')} (Remote: {job.get('is_remote',False)})\nDescription: {(job.get('description','') or '')[:1500]}"
    sm = "Score candidate fit. Return ONLY valid JSON: {\"match_score\":<0-100>,\"explanation\":\"text\",\"missing_skills\":[\"skill\"]}"
    up = f"Score candidate against job.\nResume: {resume_text[:2000]}\nJob: {jt}"
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"system","content":sm},{"role":"user","content":up}],"options":{"stream":False,"temperature":mp.get("temperature",0.2),"top_p":mp.get("top_p",0.9)}}
    r = p.chat(**cp)
    return robust_json_parse(extract_json_from_response(r["message"]["content"]))

def aggregate_jobs_from_sources(role,location="",country="worldwide",experience="",remote="",job_type="",date_posted="",page=1,limit=50,sources=None):
    if sources is None: sources = ["active_jobs_db"]
    aj = []
    if "active_jobs_db" in sources:
        res = search_jobs_active_db(role=role,location=location,country=country,experience=experience,remote=remote,job_type=job_type,date_posted=date_posted,page=page,limit=limit)
        aj.extend([normalize_active_db_job(j) for j in res.get("jobs",[])])
    if "adzuna" in sources and len(aj)<limit: aj.extend(fetch_jobs_adzuna(role=role,country=country if country!="worldwide" else "us",page=page))
    if "remoteok" in sources and (remote=="remote" or country=="remote_only"): aj.extend(fetch_jobs_remoteok(role=role))
    if "arbeitnow" in sources and len(aj)<limit: aj.extend(fetch_jobs_arbeitnow(role=role,location=location))
    if country and country!="worldwide" and country!="remote_only":
        cn = SUPPORTED_COUNTRIES.get(country,country)
        aj = [j for j in aj if cn.lower() in (j.get("country","")+j.get("location","")).lower()]
    if remote=="remote" or country=="remote_only": aj = [j for j in aj if j.get("is_remote",False)]
    if job_type and job_type in JOB_TYPES:
        jtn = JOB_TYPES[job_type].lower()
        aj = [j for j in aj if jtn in j.get("contract_type","").lower()]
    tj = len(aj); tp = max(1,(tj+limit-1)//limit); si = (page-1)*limit
    return {"jobs":aj[si:si+limit],"pagination":{"current_page":page,"total_pages":tp,"total_jobs":tj,"has_next_page":(si+limit)<tj,"has_previous_page":page>1,"limit":limit,"sources_used":sources}}

# ============================================================
# DATA MANAGEMENT
# ============================================================
def load_applications():
    f = app.config.get('APPLICATIONS_FILE','applications.json')
    if not os.path.exists(f): return []
    try:
        with open(f,encoding="utf-8") as fp: return json.load(fp)
    except: return []

def save_applications(r):
    with open(app.config.get('APPLICATIONS_FILE','applications.json'),"w",encoding="utf-8") as fp: json.dump(r,fp,indent=2,ensure_ascii=False)

# ============================================================
# BACKGROUND JOBS
# ============================================================
def set_job(jid,**kw):
    with JOBS_LOCK:
        if jid in JOBS: JOBS[jid].update(kw)

def process_job(jid,pp,fn,jd,own):
    try:
        set_job(jid,status="running",step="Reading PDF...")
        ph = PDFHandler(); rd = ph.extract_json_from_pdf(pp)
        if not is_valid_resume_data(rd): set_job(jid,status="error",error="Could not extract resume"); return
        set_job(jid,step="Checking GitHub..."); gd = {}
        pr = rd.basics.profiles if rd.basics else []
        gp = find_profile(pr,"Github")
        if gp:
            try: gd = fetch_and_display_github_info(gp.url)
            except: pass
        set_job(jid,step="Scoring..."); ev,rt = run_evaluation(rd,gd); ed = evaluation_to_dict(ev)
        cn = fn.replace(".pdf","")
        if rd.basics and rd.basics.name: cn = rd.basics.name
        res = {"id":jid,"owner":own,"candidate_name":cn,"file_name":fn,"created_at":datetime.now(timezone.utc).isoformat(),"evaluation":ed,"resume_text":rt,"resume_json":rd.model_dump() if hasattr(rd,"model_dump") else {}}
        res.update(compute_total_score(ed))
        if jd: res["job_match"]=match_job_description(rt,jd); res["job_description"]=jd
        with open(os.path.join(app.config['HISTORY_FOLDER'],f"{jid}.json"),"w",encoding="utf-8") as f: json.dump(res,f,indent=2,ensure_ascii=False)
        set_job(jid,status="done",step="Done",result=res)
    except Exception as e:
        logger.exception("Eval failed"); set_job(jid,status="error",error=str(e))

def run_ai_job_search(jid,rt,role,location="",country="worldwide",experience="",remote="",job_type="",date_posted="",page=1,limit=50,sources=None):
    try:
        set_job(jid,status="running",step="Searching jobs...")
        if sources is None: sources = ["active_jobs_db"]
        sr = aggregate_jobs_from_sources(role=role,location=location,country=country,experience=experience,remote=remote,job_type=job_type,date_posted=date_posted,page=page,limit=min(limit,100),sources=sources)
        rj = sr.get("jobs",[]); pg = sr.get("pagination",{})
        if not rj: set_job(jid,status="done",step="Done",result={"jobs":[],"pagination":pg}); return
        sj = []
        for i,j in enumerate(rj,start=1):
            set_job(jid,step=f"Scoring {i}/{len(rj)}...")
            try: sc = score_job_for_candidate(rt,j)
            except: continue
            sj.append({"job_id":j.get("job_id"),"title":j.get("title",""),"company":j.get("company",""),"location":f"{j.get('location','')}, {j.get('country','')}".strip(", "),"is_remote":j.get("is_remote",False),"apply_link":j.get("apply_link",""),"posted_at":j.get("posted_at",""),"salary":j.get("salary",""),"source":j.get("source","unknown"),"match_score":sc.get("match_score",0),"explanation":sc.get("explanation",""),"missing_skills":sc.get("missing_skills",[])})
        sj.sort(key=lambda x:x.get("match_score",0),reverse=True)
        set_job(jid,status="done",step="Done",result={"jobs":sj,"pagination":pg})
    except Exception as e:
        logger.exception("Job search failed"); set_job(jid,status="error",error=str(e))

# ============================================================
# ROUTES - AUTH
# ============================================================
@app.route("/")
def landing():
    if session.get("user_id"): return redirect(url_for("app_home"))
    return render_template("landing.html")

@app.route("/app")
@login_required_page
def app_home():
    return render_template("index.html",username=session.get("username",""))

@app.route("/signup",methods=["GET","POST"])
def signup():
    if request.method=="GET": return render_template("signup.html",error=None)
    un = request.form.get("username","").strip(); em = request.form.get("email","").strip(); pw = request.form.get("password","")
    if not un or not pw: return render_template("signup.html",error="Username and password required.")
    if len(pw)<6: return render_template("signup.html",error="Min 6 characters.")
    if find_user(un) or (em and find_user(em)): return render_template("signup.html",error="Already taken.")
    users = load_users()
    u = {"id":uuid.uuid4().hex[:12],"username":un,"email":em,"password_hash":generate_password_hash(pw),"created_at":datetime.now(timezone.utc).isoformat()}
    users.append(u); save_users(users)
    session["user_id"]=u["id"]; session["username"]=u["username"]
    return redirect(url_for("app_home"))

@app.route("/login",methods=["GET","POST"])
def login():
    if request.method=="GET": return render_template("login.html",error=None)
    idn = request.form.get("username","").strip(); pw = request.form.get("password",""); u = find_user(idn)
    if not u or not check_password_hash(u["password_hash"],pw): return render_template("login.html",error="Incorrect credentials.")
    session["user_id"]=u["id"]; session["username"]=u["username"]
    return redirect(request.args.get("next") or url_for("app_home"))

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("landing"))

# ============================================================
# ROUTES - EVALUATION
# ============================================================
@app.route("/evaluate",methods=["POST"])
@login_required_api
def evaluate():
    files = [f for f in request.files.getlist("resume") if f and f.filename and f.filename.lower().endswith(".pdf")]
    if not files: return jsonify({"success":False,"error":"Please upload PDF files"}),400
    jd = request.form.get("job_description","").strip(); own = current_user_id(); jids = []
    for file in files:
        fn = secure_filename(file.filename); jid = uuid.uuid4().hex[:12]
        pp = os.path.join(app.config["UPLOAD_FOLDER"],f"{jid}_{fn}"); file.save(pp)
        with JOBS_LOCK: JOBS[jid] = {"status":"queued","step":"Queued...","file_name":fn}
        threading.Thread(target=process_job,args=(jid,pp,fn,jd,own),daemon=True).start()
        jids.append(jid)
    return jsonify({"success":True,"job_ids":jids})

@app.route("/status/<jid>")
@login_required_api
def status(jid):
    with JOBS_LOCK: j = JOBS.get(jid)
    if not j: return jsonify({"success":False,"error":"Unknown job"}),404
    return jsonify(j)

@app.route("/history")
@login_required_api
def history():
    own = current_user_id(); recs = []
    for fn in sorted(os.listdir(app.config['HISTORY_FOLDER']),reverse=True):
        if not fn.endswith(".json"): continue
        try:
            with open(os.path.join(app.config['HISTORY_FOLDER'],fn),encoding="utf-8") as f: d = json.load(f)
            if d.get("owner")!=own: continue
            recs.append({"id":d.get("id"),"candidate_name":d.get("candidate_name"),"created_at":d.get("created_at"),"total_score":d.get("total_score"),"max_score":d.get("max_score"),"has_job_match":"job_match" in d})
        except: pass
    return jsonify({"success":True,"data":recs})

@app.route("/history/<rid>")
@login_required_api
def history_detail(rid):
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    if not os.path.exists(p): return jsonify({"success":False,"error":"Not found"}),404
    with open(p,encoding="utf-8") as f: d = json.load(f)
    if d.get("owner")!=current_user_id(): return jsonify({"success":False,"error":"Not found"}),404
    return jsonify({"success":True,"data":d})

@app.route("/history/<rid>",methods=["DELETE"])
@login_required_api
def history_delete(rid):
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    if os.path.exists(p):
        with open(p,encoding="utf-8") as f: d = json.load(f)
        if d.get("owner")==current_user_id(): os.remove(p)
    return jsonify({"success":True,"deleted":True})

@app.route("/export/<rid>.pdf")
@login_required_api
def export_pdf(rid):
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    if not os.path.exists(p): return jsonify({"success":False,"error":"Not found"}),404
    with open(p,encoding="utf-8") as f: rec = json.load(f)
    if rec.get("owner")!=current_user_id(): return jsonify({"success":False,"error":"Not found"}),404
    op = os.path.join(app.config['HISTORY_FOLDER'],f"{rid}_report.pdf"); build_pdf_report(rec,op)
    sn = re.sub(r"[^A-Za-z0-9_-]+","_",rec.get("candidate_name","candidate"))
    return send_file(op,as_attachment=True,download_name=f"{sn}_evaluation.pdf")

@app.route("/interview_questions/<rid>",methods=["POST"])
@login_required_api
def interview_questions(rid):
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    if not os.path.exists(p): return jsonify({"success":False,"error":"Not found"}),404
    with open(p,encoding="utf-8") as f: rec = json.load(f)
    if rec.get("owner")!=current_user_id(): return jsonify({"success":False,"error":"Not found"}),404
    rt = rec.get("resume_text")
    if not rt: return jsonify({"success":False,"error":"No resume text"}),422
    try:
        qs = generate_interview_questions(rt); rec["interview_questions"]=qs
        with open(p,"w",encoding="utf-8") as f: json.dump(rec,f,indent=2,ensure_ascii=False)
        return jsonify({"success":True,"data":{"interview_questions":qs}})
    except Exception as e: return jsonify({"success":False,"error":str(e)}),500

# ============================================================
# ROUTES - APPLICATIONS
# ============================================================
@app.route("/applications",methods=["GET"])
@login_required_api
def list_applications():
    own = current_user_id()
    return jsonify({"success":True,"data":[a for a in load_applications() if a.get("owner")==own]})

@app.route("/applications",methods=["POST"])
@login_required_api
def add_application():
    d = request.get_json(force=True) or {}
    if not d.get("candidate_name") or not d.get("company"): return jsonify({"success":False,"error":"Name and company required"}),400
    recs = load_applications()
    rec = {"id":uuid.uuid4().hex[:12],"owner":current_user_id(),"candidate_id":d.get("candidate_id",""),"candidate_name":d.get("candidate_name",""),"company":d.get("company",""),"role":d.get("role",""),"status":d.get("status","Applied"),"date_applied":d.get("date_applied") or datetime.now(timezone.utc).date().isoformat(),"link":d.get("link",""),"notes":d.get("notes",""),"created_at":datetime.now(timezone.utc).isoformat()}
    recs.append(rec); save_applications(recs)
    return jsonify({"success":True,"data":rec})

@app.route("/applications/<aid>",methods=["PATCH"])
@login_required_api
def update_application(aid):
    d = request.get_json(force=True) or {}; recs = load_applications()
    for r in recs:
        if r["id"]==aid and r.get("owner")==current_user_id():
            for f in ("status","role","company","link","notes","date_applied"):
                if f in d: r[f]=d[f]
            save_applications(recs); return jsonify({"success":True,"data":r})
    return jsonify({"success":False,"error":"Not found"}),404

@app.route("/applications/<aid>",methods=["DELETE"])
@login_required_api
def delete_application(aid):
    own = current_user_id(); recs = load_applications()
    recs = [r for r in recs if not (r["id"]==aid and r.get("owner")==own)]
    save_applications(recs); return jsonify({"success":True,"deleted":True})

# ============================================================
# ROUTES - JOB SEARCH
# ============================================================
@app.route("/job_search/start",methods=["POST"])
@login_required_api
def job_search_start():
    d = request.get_json(force=True) or {}
    cid = d.get("candidate_id",""); role = d.get("role","").strip(); loc = d.get("location","").strip()
    country = d.get("country","worldwide"); exp = d.get("experience",""); remote = d.get("remote","")
    jt = d.get("job_type",""); dp = d.get("date_posted","")
    page = max(1,int(d.get("page",1))); limit = min(100,max(10,int(d.get("limit",50))))
    sources = d.get("sources",["active_jobs_db"])
    if not cid: return jsonify({"success":False,"error":"Select a candidate"}),400
    if not role: return jsonify({"success":False,"error":"Enter a role"}),400
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(cid)}.json")
    if not os.path.exists(p): return jsonify({"success":False,"error":"Not found"}),404
    with open(p,encoding="utf-8") as f: rec = json.load(f)
    if rec.get("owner")!=current_user_id(): return jsonify({"success":False,"error":"Not found"}),404
    rt = rec.get("resume_text")
    if not rt: return jsonify({"success":False,"error":"No resume text"}),422
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK: JOBS[jid] = {"status":"queued","step":"Queued..."}
    threading.Thread(target=run_ai_job_search,args=(jid,rt,role,loc,country,exp,remote,jt,dp,page,limit,sources),daemon=True).start()
    return jsonify({"success":True,"job_id":jid})

@app.route("/job_search/filters",methods=["GET"])
@login_required_api
def get_search_filters():
    return jsonify({"success":True,"data":{"countries":SUPPORTED_COUNTRIES,"experience_levels":EXPERIENCE_FILTER_MAP,"work_arrangements":WORK_ARRANGEMENT_MAP,"job_types":JOB_TYPES,"date_posted_filters":DATE_POSTED_FILTERS}})

@app.route("/job_search/save",methods=["POST"])
@login_required_api
def job_search_save():
    d = request.get_json(force=True) or {}; job = d.get("job") or {}
    cn = d.get("candidate_name",""); cid = d.get("candidate_id","")
    if not job.get("company") or not cn: return jsonify({"success":False,"error":"Missing info"}),400
    recs = load_applications()
    rec = {"id":uuid.uuid4().hex[:12],"owner":current_user_id(),"candidate_id":cid,"candidate_name":cn,"company":job.get("company",""),"role":job.get("title",""),"status":"Saved","date_applied":datetime.now(timezone.utc).date().isoformat(),"link":job.get("apply_link",""),"notes":f"AI match: {job.get('match_score','-')}%","match_score":job.get("match_score"),"source":f"ai_search","created_at":datetime.now(timezone.utc).isoformat()}
    recs.append(rec); save_applications(recs)
    return jsonify({"success":True,"data":rec})

# ============================================================
# ROUTES - PROFILE
# ============================================================
@app.route("/me")
@login_required_api
def me():
    own = current_user_id(); users = load_users()
    u = next((u for u in users if u["id"]==own),None)
    if not u: return jsonify({"success":False,"error":"Not found"}),404
    ec = sum(1 for fn in os.listdir(app.config['HISTORY_FOLDER']) if fn.endswith(".json") and json.load(open(os.path.join(app.config['HISTORY_FOLDER'],fn))).get("owner")==own)
    ac = len([a for a in load_applications() if a.get("owner")==own])
    return jsonify({"success":True,"data":{"username":u["username"],"email":u.get("email",""),"created_at":u.get("created_at"),"evaluations_count":ec,"applications_count":ac}})

@app.route("/change_password",methods=["POST"])
@login_required_api
def change_password():
    d = request.get_json(force=True) or {}
    cp = d.get("current_password",""); np = d.get("new_password","")
    if len(np)<6: return jsonify({"success":False,"error":"Min 6 chars"}),400
    users = load_users(); own = current_user_id()
    u = next((u for u in users if u["id"]==own),None)
    if not u or not check_password_hash(u["password_hash"],cp): return jsonify({"success":False,"error":"Wrong password"}),400
    u["password_hash"]=generate_password_hash(np); save_users(users)
    return jsonify({"success":True,"message":"Updated"})

# ============================================================
# SEO & LEGAL
# ============================================================
@app.route('/privacy')
def privacy(): return render_template('privacy.html')

@app.route('/terms')
def terms(): return render_template('terms.html')

@app.route('/sitemap.xml')
def sitemap():
    bu = os.getenv('APP_URL','https://hiringagent.onrender.com')
    pages = [{'url':'/','cf':'daily','pr':'1.0'},{'url':'/signup','cf':'monthly','pr':'0.9'},{'url':'/login','cf':'monthly','pr':'0.8'},{'url':'/app','cf':'weekly','pr':'0.7'},{'url':'/privacy','cf':'yearly','pr':'0.3'},{'url':'/terms','cf':'yearly','pr':'0.3'}]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for p in pages: xml += f'  <url>\n    <loc>{bu}{p["url"]}</loc>\n    <changefreq>{p["cf"]}</changefreq>\n    <priority>{p["pr"]}</priority>\n  </url>\n'
    xml += '</urlset>'
    r = make_response(xml); r.headers['Content-Type']='application/xml'; return r

@app.route('/robots.txt')
def robots():
    bu = os.getenv('APP_URL','https://hiringagent.onrender.com')
    rt = f'User-agent: *\nAllow: /\nDisallow: /uploads/\nDisallow: /evaluations/\nCrawl-delay: 1\nSitemap: {bu}/sitemap.xml'
    r = make_response(rt); r.headers['Content-Type']='text/plain'; return r

@app.route('/health')
def health_check():
    return jsonify({"status":"healthy","timestamp":datetime.now(timezone.utc).isoformat(),"version":"3.0.0"})

# ============================================================
# FEATURE 1: CODING ASSESSMENT
# ============================================================
@app.route("/coding/generate",methods=["POST"])
@login_required_api
def generate_coding_test():
    d = request.get_json(force=True) or {}
    rid = d.get("record_id",""); diff = d.get("difficulty","medium"); lang = d.get("language","")
    if not rid: return jsonify({"success":False,"error":"Select a candidate"}),400
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    if not os.path.exists(p): return jsonify({"success":False,"error":"Not found"}),404
    with open(p,encoding="utf-8") as f: rec = json.load(f)
    rt = rec.get("resume_text","")
    if not rt: return jsonify({"success":False,"error":"No resume text"}),422
    pr = initialize_llm_provider(DEFAULT_MODEL)
    sm = "Create coding tests. Return ONLY valid JSON, no markdown."
    up = f"""Generate {diff} coding test. Language: {lang or 'relevant'}. Return:
{{"test_title":"title","duration_minutes":45,"difficulty":"{diff}","language":"{lang or 'auto'}","questions":[{{"id":1,"title":"Q","description":"desc","example":{{"input":"in","output":"out"}},"test_cases":[{{"input":"in","expected_output":"out"}}],"hints":["hint"]}}],"evaluation_criteria":{{"correctness":40,"code_quality":30,"efficiency":20,"edge_cases":10}}}}
Resume: {rt[:2000]}"""
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"system","content":sm},{"role":"user","content":up}],"options":{"stream":False,"temperature":0.3,"top_p":0.9}}
    try:
        r = pr.chat(**cp); td = robust_json_parse(extract_json_from_response(r["message"]["content"]))
        return jsonify({"success":True,"data":td})
    except Exception as e: return jsonify({"success":False,"error":str(e)}),500

@app.route("/coding/evaluate",methods=["POST"])
@login_required_api
def evaluate_coding_submission():
    d = request.get_json(force=True) or {}
    q = d.get("question",""); code = d.get("code",""); lang = d.get("language","python")
    if not code: return jsonify({"success":False,"error":"No code"}),400
    pr = initialize_llm_provider(DEFAULT_MODEL)
    sm = "Code reviewer. Return ONLY valid JSON."
    up = f"""Evaluate {lang} code. Return:
{{"score":<0-100>,"feedback":"text","strengths":["s"],"improvements":["i"],"suggested_solution":"text"}}
Q: {q}\nCode: {code[:2000]}"""
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"system","content":sm},{"role":"user","content":up}],"options":{"stream":False,"temperature":0.1,"top_p":0.9}}
    try:
        r = pr.chat(**cp); ev = robust_json_parse(extract_json_from_response(r["message"]["content"]))
        return jsonify({"success":True,"data":ev})
    except Exception as e: return jsonify({"success":False,"error":str(e)}),500

# ============================================================
# FEATURE 2: NETWORKING
# ============================================================
@app.route("/networking/find-recruiters",methods=["POST"])
@login_required_api
def find_recruiters():
    d = request.get_json(force=True) or {}
    rid = d.get("record_id",""); role = d.get("role","").strip(); loc = d.get("location","").strip()
    if not rid or not role: return jsonify({"success":False,"error":"Candidate and role required"}),400
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    if not os.path.exists(p): return jsonify({"success":False,"error":"Not found"}),404
    with open(p,encoding="utf-8") as f: rec = json.load(f)
    rt = rec.get("resume_text",""); pr = initialize_llm_provider(DEFAULT_MODEL)
    sm = "Networking assistant. Return ONLY valid JSON."
    up = f"""Generate networking strategy. Return:
{{"linkedin_search_queries":["q1"],"target_companies":["c1"],"connection_message_templates":[{{"scenario":"s","subject":"s","message":"m"}}],"networking_tips":["tip"],"hashtags_to_follow":["#tag"]}}
Candidate: {rt[:2000]}\nRole: {role}\nLocation: {loc or 'Any'}"""
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"system","content":sm},{"role":"user","content":up}],"options":{"stream":False,"temperature":0.4,"top_p":0.9}}
    try:
        r = pr.chat(**cp); nd = robust_json_parse(extract_json_from_response(r["message"]["content"]))
        return jsonify({"success":True,"data":nd})
    except Exception as e: return jsonify({"success":False,"error":str(e)}),500

@app.route("/networking/generate-message",methods=["POST"])
@login_required_api
def generate_connection_message():
    d = request.get_json(force=True) or {}
    rid = d.get("record_id",""); rn = d.get("recruiter_name","Hiring Manager")
    rc = d.get("recruiter_company","company"); jt = d.get("job_title","")
    if not rid: return jsonify({"success":False,"error":"Candidate required"}),400
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    with open(p,encoding="utf-8") as f: rec = json.load(f)
    rt = rec.get("resume_text",""); cn = rec.get("candidate_name","Candidate")
    pr = initialize_llm_provider(DEFAULT_MODEL)
    sm = "LinkedIn expert. Return ONLY valid JSON."
    up = f"""Generate LinkedIn message from {cn} to {rn} at {rc}. Return:
{{"messages":[{{"style":"professional","message":"text"}},{{"style":"friendly","message":"text"}},{{"style":"direct","message":"text"}}],"follow_up_tips":["tip"]}}
Background: {rt[:1500]}\nInterest: {jt or 'opportunities'}"""
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"system","content":sm},{"role":"user","content":up}],"options":{"stream":False,"temperature":0.7,"top_p":0.9}}
    try:
        r = pr.chat(**cp); ms = robust_json_parse(extract_json_from_response(r["message"]["content"]))
        return jsonify({"success":True,"data":ms})
    except Exception as e: return jsonify({"success":False,"error":str(e)}),500

# ============================================================
# FEATURE 3: MOCK INTERVIEW
# ============================================================
@app.route("/interview/start",methods=["POST"])
@login_required_api
def start_mock_interview():
    d = request.get_json(force=True) or {}
    rid = d.get("record_id",""); itype = d.get("type","technical"); diff = d.get("difficulty","medium")
    if not rid: return jsonify({"success":False,"error":"Select a candidate"}),400
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    with open(p,encoding="utf-8") as f: rec = json.load(f)
    rt = rec.get("resume_text",""); cn = rec.get("candidate_name","Candidate")
    pr = initialize_llm_provider(DEFAULT_MODEL)
    sm = "Interviewer. Return ONLY valid JSON, no markdown."
    up = f"""Generate {diff} {itype} interview for {cn}. Return:
{{"interview_id":"mock-{uuid.uuid4().hex[:8]}","candidate_name":"{cn}","type":"{itype}","difficulty":"{diff}","duration_estimate":"30 min","questions":[{{"id":1,"category":"technical","question":"Question?","expected_topics":["topic"],"follow_up":"Follow-up?"}}],"scoring_rubric":{{"technical_accuracy":30,"communication":25,"problem_solving":25,"experience_depth":20}}}}
Resume: {rt[:1500]}"""
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"system","content":sm},{"role":"user","content":up}],"options":{"stream":False,"temperature":0.3,"top_p":0.9}}
    try:
        r = pr.chat(**cp); iv = robust_json_parse(extract_json_from_response(r["message"]["content"]))
        return jsonify({"success":True,"data":iv})
    except Exception as e: return jsonify({"success":False,"error":str(e)}),500

@app.route("/interview/evaluate-answer",methods=["POST"])
@login_required_api
def evaluate_interview_answer():
    d = request.get_json(force=True) or {}
    q = d.get("question",""); a = d.get("answer",""); cat = d.get("category","technical")
    if not a: return jsonify({"success":False,"error":"No answer"}),400
    pr = initialize_llm_provider(DEFAULT_MODEL)
    up = f"""Evaluate answer. Return:
{{"overall_score":<0-100>,"strengths":["s"],"areas_to_improve":["a"],"feedback_summary":"text","tips":["tip"]}}
Q: {q}\nA: {a[:1500]}"""
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"user","content":up}],"options":{"stream":False,"temperature":0.3,"top_p":0.9}}
    try:
        r = pr.chat(**cp); ev = robust_json_parse(extract_json_from_response(r["message"]["content"]))
        return jsonify({"success":True,"data":ev})
    except Exception as e: return jsonify({"success":False,"error":str(e)}),500

# ============================================================
# FEATURE 4: COVER LETTER
# ============================================================
@app.route("/cover-letter/generate",methods=["POST"])
@login_required_api
def generate_cover_letter():
    d = request.get_json(force=True) or {}
    rid = d.get("record_id",""); jt = d.get("job_title","").strip(); comp = d.get("company_name","").strip()
    jd = d.get("job_description","").strip()
    if not rid or not jt or not comp: return jsonify({"success":False,"error":"Candidate, title, company required"}),400
    p = os.path.join(app.config['HISTORY_FOLDER'],f"{secure_filename(rid)}.json")
    if not os.path.exists(p): return jsonify({"success":False,"error":"Not found"}),404
    with open(p,encoding="utf-8") as f: rec = json.load(f)
    rt = rec.get("resume_text",""); cn = rec.get("candidate_name","Candidate")
    pr = initialize_llm_provider(DEFAULT_MODEL)
    sm = "Cover letter writer. Return ONLY valid JSON."
    up = f"""Generate cover letter for {cn} - {jt} at {comp}. Return:
{{"candidate_name":"{cn}","job_title":"{jt}","company":"{comp}","cover_letter":"Full letter text","key_skills_highlighted":["skill"],"subject_line":"Subject","tips":["tip"]}}
Resume: {rt[:2000]}\nJob: {jd[:1000] or 'Not provided'}"""
    cp = {"model":DEFAULT_MODEL,"messages":[{"role":"system","content":sm},{"role":"user","content":up}],"options":{"stream":False,"temperature":0.7,"top_p":0.9}}
    try:
        r = pr.chat(**cp); cl = robust_json_parse(extract_json_from_response(r["message"]["content"]))
        return jsonify({"success":True,"data":cl})
    except Exception as e: return jsonify({"success":False,"error":str(e)}),500

# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success":False,"error":"Not found"}),404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500: {e}")
    return jsonify({"success":False,"error":"Server error"}),500

# ============================================================
# PDF BUILDER
# ============================================================
def build_pdf_report(rec,op):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("H1",parent=ss["Title"],fontSize=20,spaceAfter=4)
    sb = ParagraphStyle("Sub",parent=ss["Normal"],textColor=colors.grey,spaceAfter=14)
    h2 = ParagraphStyle("H2",parent=ss["Heading2"],spaceBefore=14,spaceAfter=6)
    bd = ParagraphStyle("Body",parent=ss["Normal"],fontSize=10,leading=14)
    st = [Paragraph(rec.get("candidate_name","Candidate"),h1),Paragraph(f"Report &middot; {rec.get('created_at','')[:19].replace('T',' ')} UTC",sb),Paragraph(f"<b>Score: {rec.get('total_score')}/{rec.get('max_score')}</b>",ss["Heading2"]),Spacer(1,10)]
    ev = rec.get("evaluation") or {}; sc = ev.get("scores") or {}
    td = [["Category","Score","Max","Evidence"]]
    for k,fm in CATEGORY_MAX.items():
        c = sc.get(k) or {}; mv = c.get("max",fm)
        td.append([k.replace("_"," ").title(),str(c.get("score","-")),str(mv),Paragraph(c.get("evidence","") or "-",bd)])
    tb = Table(td,colWidths=[1.2*inch,0.6*inch,0.5*inch,3.7*inch])
    tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#222831")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),9),("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5f5")])]))
    st.append(tb)
    SimpleDocTemplate(op,pagesize=letter,topMargin=0.7*inch,bottomMargin=0.7*inch).build(st)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT",5000))
    debug = os.getenv("FLASK_ENV","development")=="development"
    app.run(host="0.0.0.0",port=port,debug=debug)