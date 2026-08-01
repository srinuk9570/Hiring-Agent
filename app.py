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

def robust_json_parse(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?```\s*', '', cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except json.JSONDecodeError:
            pass
    fixed = re.sub(r',\s*}', '}', cleaned)
    fixed = re.sub(r',\s*]', ']', fixed)
    fixed = fixed.replace("'", '"')
    s = fixed.find('{'); e = fixed.rfind('}')
    if s != -1 and e != -1: fixed = fixed[s:e+1]
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    logger.warning("All JSON parse strategies failed, using manual extraction")
    return _manual_extract(text)

def _manual_extract(text: str) -> dict:
    result = {}
    for key in ['match_score','score','overall_score','technical_accuracy','communication_clarity','confidence_level','correctness_score','code_quality_score','efficiency_score','edge_cases_score','filler_words_count','word_count','duration_minutes']:
        m = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', text)
        if m:
            try: result[key] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
            except: result[key] = m.group(1)
    for key in ['test_title','title','candidate_name','type','difficulty','language','feedback','feedback_summary','explanation','suggested_solution','suggested_answer_outline','subject_line','cover_letter','why_this_company','duration_estimate','time_complexity','space_complexity','pace','structure','overall_assessment','interview_id','job_title','company','job_id']:
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if m: result[key] = m.group(1)
    for key in ['strengths','improvements','missing_skills','tips','areas_to_improve','questions','cover_letters','messages','linkedin_search_queries','recruiter_types','target_companies','networking_tips','hashtags_to_follow','key_skills_highlighted','follow_up_tips','hints','test_cases','expected_topics','good_answer_indicators','key_phrases_used','improvement_suggestions','filler_words','coding_questions','core_subject_questions','project_questions','behavioral_questions']:
        arr_match = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if arr_match:
            items = re.findall(r'"([^"]*)"', arr_match.group(1))
            if items: result[key] = items
    if 'match_score' not in result: result['match_score'] = 50
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
# JOB SEARCH (kept same - omitted for brevity)
# ============================================================
# [All job search functions remain the same]

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
        set_job(jid,step="Analyzing resume with AI..."); gd = {}
        pr = rd.basics.profiles if rd.basics else []
        gp = find_profile(pr,"Github")
        if gp:
            try: gd = fetch_and_display_github_info(gp.url)
            except: pass
        set_job(jid,step="Scoring resume..."); ev,rt = run_evaluation(rd,gd); ed = evaluation_to_dict(ev)
        cn = fn.replace(".pdf","")
        if rd.basics and rd.basics.name: cn = rd.basics.name
        res = {"id":jid,"owner":own,"candidate_name":cn,"file_name":fn,"created_at":datetime.now(timezone.utc).isoformat(),"evaluation":ed,"resume_text":rt,"resume_json":rd.model_dump() if hasattr(rd,"model_dump") else {}}
        res.update(compute_total_score(ed))
        if jd: res["job_match"]=match_job_description(rt,jd); res["job_description"]=jd
        with open(os.path.join(app.config['HISTORY_FOLDER'],f"{jid}.json"),"w",encoding="utf-8") as f: json.dump(res,f,indent=2,ensure_ascii=False)
        set_job(jid,status="done",step="Done",result=res)
    except Exception as e:
        logger.exception("Eval failed"); set_job(jid,status="error",error=str(e))

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
    # FIX: Return pending instead of 404
    if not j: return jsonify({"status":"pending","step":"Initializing..."})
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

# [All other routes remain the same - applications, job search, profile, SEO, features]

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
# PDF BUILDER & MAIN
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
    bd = ParagraphStyle("Body",parent=ss["Normal"],fontSize=10,leading=14)
    st = [Paragraph(rec.get("candidate_name","Candidate"),h1),Paragraph(f"Report - {rec.get('created_at','')[:19].replace('T',' ')} UTC",sb),Paragraph(f"<b>Score: {rec.get('total_score')}/{rec.get('max_score')}</b>",ss["Heading2"]),Spacer(1,10)]
    ev = rec.get("evaluation") or {}; sc = ev.get("scores") or {}
    td = [["Category","Score","Max","Evidence"]]
    for k,fm in CATEGORY_MAX.items():
        c = sc.get(k) or {}; mv = c.get("max",fm)
        td.append([k.replace("_"," ").title(),str(c.get("score","-")),str(mv),Paragraph(c.get("evidence","") or "-",bd)])
    tb = Table(td,colWidths=[1.2*inch,0.6*inch,0.5*inch,3.7*inch])
    tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#222831")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),9),("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5f5")])]))
    st.append(tb)
    SimpleDocTemplate(op,pagesize=letter,topMargin=0.7*inch,bottomMargin=0.7*inch).build(st)

if __name__ == "__main__":
    port = int(os.getenv("PORT",5000))
    debug = os.getenv("FLASK_ENV","development")=="development"
    app.run(host="0.0.0.0",port=port,debug=debug)