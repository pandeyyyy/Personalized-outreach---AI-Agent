#!/usr/bin/env python3
"""
Prospect -> researched, verified, human-reviewed outreach draft.

Pipeline
  1. GATHER   (separate research buckets) person news, employment/role-change search, company news,
              company site (home/about/team/product/careers/blog/news), optional pasted LinkedIn/notes
  2. ANALYSE  Gemini checks identity + employment status, extracts stable company/founder FACTS
              and candidate news SIGNALS (each tied to evidence ids)
  3. JUDGE    deterministic scoring + safety gates, then picks a PERSONALISATION LEVEL:
                 3 Personal         strong recent, evidence-backed hook
                 2 Founder/Company  no strong news, but stable public company/founder facts exist
                 1 Role-based       very little public info (e.g. tiny early-stage startup, or identity
                                    can't be verified) -> short, generic, positive role note, flagged
                                    for human verification
                 0 Block/Hold       prospect moved and new role can't be found, wrong input that can't be
                                    corrected, or a death/tragedy (no draft is written at all)
  4. DRAFT    Gemini writes from the chosen evidence only, under strict rules
  5. VERIFY   second Gemini pass: every factual claim must be backed by evidence (+ lint)
  6. REVIEW   human approves / edits / gets a NEW draft / skips. Nothing is ever sent by this script.

Edge cases handled:
  * Prospect moved: we extract their NEW company + role, research the new company, and draft an outreach
    for the new role (flagged for human verification). Optionally identifies a SUCCESSOR at the old company.
  * Wrong input (e.g. Deepinder Goel + Swiggy): evidence shows the person belongs to a DIFFERENT company ->
    we find the real company, research it and draft for it (flagged NEEDS_REVIEW, no "moved" claim).
  * Negative news (e.g. Byju Raveendran): detected by LLM sensitivity labels PLUS a deterministic keyword
    scan of person/company headlines. It is IGNORED: removed from the evidence, and the email is written
    from positive/neutral facts only (or the generic note if nothing positive exists) without mentioning
    it. Only deaths/tragedies still HOLD (no draft).
  * Small startups / name clashes: evidence must actually mention the company (or come from its own site /
    rep notes); otherwise a short, generic, positive Level-1 note with no company claims.

Every result carries an EXPLANATION: person status, level, main reason, recommended action.

Setup:  pip install requests beautifulsoup4
        export GEMINI_API_KEY=...   (free key: https://aistudio.google.com/apikey)
"""
import argparse, csv, json, os, re, subprocess, sys, tempfile, time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

_POOL = ThreadPoolExecutor(max_workers=4)


def _get_many(urls, headers, connect_read_timeout=6, hard_wall_timeout=10):
    """Fetch several URLs in parallel. requests' own timeout can be bypassed by OS-level network
    stalls (seen on some Windows/Wi-Fi setups). Each call runs in a worker thread and is abandoned
    if it doesn't return in time, so it can never hang the app. Returns a list (None = failed)."""
    futs = [_POOL.submit(requests.get, u, headers=headers,
                         timeout=(connect_read_timeout, connect_read_timeout)) for u in urls]
    out = []
    for f in futs:
        try:
            out.append(f.result(timeout=hard_wall_timeout))
        except Exception:
            out.append(None)
    return out


MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MIN_INTERVAL = 6.5          # free tier is ~10 requests/min; stay under it
SCORE_THRESHOLD = 5.5       # below this, a news signal is not worth personalising on
MAX_STALE_DAYS = 365
MAX_WORDS = 120
FALLBACK_BODY_WORDS = 70    # low-info prospects get a deliberately short, generic note
OUT_DIR = Path("out")
UA = {"User-Agent": "Mozilla/5.0 (compatible; OutreachResearchBot/1.0)"}

LEVEL_NAMES = {3: "Personal", 2: "Founder/Company", 1: "Role-based", 0: "Block"}
FACT_CORE = {"what_company_does", "product", "customers", "industry"}   # needed for level 2
FACT_CATEGORIES = ["what_company_does", "product", "customers", "industry", "founder_role",
                   "activity", "hiring"]

DEFAULT_SELLER = {
    "company": "FlowMetrics",
    "what_we_sell": "Revenue analytics platform that gives sales leaders pipeline forecasting "
                    "accuracy and rep-level coaching insights.",
    "typical_pains": ["forecast misses", "ramping new reps slowly", "scaling a sales team",
                      "no visibility into pipeline quality"],
    "sender_name": "Alex",
    "cta_style": "a soft, low-friction question (not a meeting request)",
    "purpose": "Start a conversation about whether our product could help their team",
    "tone": "friendly and professional",
}

BANNED = ["i hope this finds you well", "i came across your", "reaching out to", "synergy",
          "circle back", "touch base", "game-changer", "revolutionary", "quick call",
          "pick your brain", "just checking in", "i noticed you're struggling"]


def get_purpose(seller):
    """The single source of truth for the email's purpose (falls back to the default)."""
    return (seller.get("purpose") or "").strip() or DEFAULT_SELLER["purpose"]


# --------------------------------------------------------------------------- safety / relevance helpers
NEGATIVE_TERMS = [
    "fraud", "scam", "arrested", "arrest", "lawsuit", "sued", "insolvency", "bankruptcy", "bankrupt",
    "default", "defaulted", "probe", "raid", "raids", "money laundering", "fema", "layoffs", "lays off",
    "laid off", "shuts down", "shutdown", "data breach", "breach", "hacked", "penalty", "fined",
    "crackdown", "banned", "scandal", "controversy", "allegations", "accused", "nclt", "unpaid",
    "dues", "cheating", "non-compliance", "investigation", "summons", "fugitive",
]
# Deaths / tragedies: a cold email is tasteless here, so these still HOLD (no draft at all).
# Everything in NEGATIVE_TERMS is simply ignored and the email is written from positive facts.
GRAVE_TERMS = ["dies", "dead", "death", "passed away", "demise", "killed", "tragedy", "suicide"]
_COMPANY_SUFFIX = r"\b(pvt|private|ltd|limited|inc|llc|technologies|technology|labs|corp|co)\b"


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def _company_key(name):
    return _norm(re.sub(_COMPANY_SUFFIX, " ", (name or "").lower()))


def mentions_company(e, p):
    """True if this evidence is really about the prospect's company: its own site, rep notes,
    or an article whose title/text names the company. Stops namesake companies leaking in
    (very common for small startups)."""
    if e["kind"].startswith("site") or e["kind"] == "user_notes":
        return True
    key = _company_key(p["company"])
    return bool(key) and key in _norm(e["title"] + " " + e["text"])


def negative_news_hits(p, ev, terms=None):
    """Deterministic detector: person/company headlines that name the person or company AND
    contain a negative term (or the given terms). Independent of the LLM, so it can't be 'missed'."""
    terms = terms or NEGATIVE_TERMS
    name, comp = _norm(p["name"]), _company_key(p["company"])
    hits = []
    for e in ev:
        if e.get("category") not in ("person_news", "company_news"):
            continue
        t = _norm(e["title"])
        if not (name in t or (comp and comp in t)):
            continue
        if any(re.search(r"\b" + re.escape(_norm(w)) + r"\b", t) for w in terms):
            hits.append(e)
    return hits


def _has_term(text, terms):
    t = _norm(text)
    return any(re.search(r"\b" + re.escape(_norm(w)) + r"\b", t) for w in terms)


def hold_for_grave(packet, hits):
    """Deaths / tragedies only: no draft at all."""
    top = hits[0]
    packet["status"] = "HOLD"
    packet["level"] = 0
    packet["mode"] = None
    packet["flags"].append(f"GRAVE EVENT detected ({len(hits)} item(s)), e.g. '{top['title'][:110]}' "
                           f"({top['date'] or 'undated'}). No email drafted. Do not send now; "
                           "re-evaluate later or use a human-led approach.")


def suppress_negative(packet, p, ev):
    """Negative news is IGNORED, not used: it is removed from the evidence the model sees, so the email is
    built only from positive/neutral facts and never mentions it. Deaths/tragedies still HOLD.
    Returns (clean_evidence, is_grave)."""
    grave = negative_news_hits(p, ev, GRAVE_TERMS)
    if grave:
        hold_for_grave(packet, grave)
        return ev, True
    neg = negative_news_hits(p, ev)
    if not neg:
        return ev, False
    drop = {e["id"] for e in neg}
    packet["suppressed_negative"] += [e["title"] for e in neg]
    packet["negative_ctx"] = packet.get("negative_ctx") or neg[0]["title"]
    packet["flags"].append(f"NEGATIVE NEWS FOUND AND IGNORED ({len(neg)} item(s)), e.g. '{neg[0]['title'][:110]}'. "
                           "The draft uses positive/neutral facts only and does not mention it.")
    return [e for e in ev if e["id"] not in drop], False


def negative_leak(draft):
    """Lint: when negative news exists, the draft must not contain negative wording."""
    low = _norm(draft["subject"] + " " + draft["body"])
    return [f"mentions negative term: '{w}'" for w in NEGATIVE_TERMS + GRAVE_TERMS
            if re.search(r"\b" + re.escape(_norm(w)) + r"\b", low)]


# --------------------------------------------------------------------------- progress hook
# The UI sets STAGE_HOOK to a callable(stage_name) so it can show which pipeline step is running.
# Stage names: Gather, Analyse, Judge, Draft, Verify.
STAGE_HOOK = None


def stage(name):
    print(f"[{name}]")
    if STAGE_HOOK:
        try:
            STAGE_HOOK(name)
        except Exception:
            pass


# --------------------------------------------------------------------------- Gemini
_last_call = 0.0


def gemini(system, prompt, temperature=0.3):
    """Call Gemini REST API, return parsed JSON. Handles rate limits + retries."""
    global _last_call
    if not API_KEY:
        sys.exit("Set GEMINI_API_KEY first (free: https://aistudio.google.com/apikey)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    gen_cfg = {"temperature": temperature, "responseMimeType": "application/json"}
    if "2.5" in MODEL:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}   # faster, cheaper, fine for extraction
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg}
    for attempt in range(5):
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = requests.post(url, json=body, headers={"x-goog-api-key": API_KEY}, timeout=90)
        except requests.RequestException as e:
            print(f"  [gemini] network error: {e}; retrying"); time.sleep(3 * (attempt + 1)); continue
        if r.status_code in (429, 500, 503):
            print(f"  [gemini] {r.status_code}; backing off"); time.sleep(10 * (attempt + 1)); continue
        if r.status_code != 200:
            sys.exit(f"Gemini error {r.status_code}: {r.text[:400]}")
        try:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            parsed = json.loads(text)
            if isinstance(parsed, list):          # model sometimes wraps the object in a list
                parsed = next((x for x in parsed if isinstance(x, dict)), None)
            if not isinstance(parsed, dict):
                raise ValueError("response was not a JSON object")
            return parsed
        except (KeyError, IndexError, json.JSONDecodeError, ValueError):
            print("  [gemini] unparseable response; retrying"); continue
    raise RuntimeError("Gemini failed after retries")


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- 1. GATHER
def _iso(dt):
    return dt.astimezone(timezone.utc).date().isoformat() if dt else None


def fetch_news(query, limit=8, window="365d", category="company_news"):
    url = (f"https://news.google.com/rss/search?q={quote_plus(query + ' when:' + window)}"
           f"&hl=en-US&gl=US&ceid=US:en")
    try:
        r = requests.get(url, headers=UA, timeout=12)
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"  [news] failed for '{query}': {e}")
        return []
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:
            dt = None
        items.append({"kind": "news", "category": category, "title": title, "text": title,
                      "url": it.findtext("link"), "date": _iso(dt)})
        if len(items) >= limit:
            break
    return items


# (path, label, research category)
SITE_PAGES = [("", "homepage", "company"), ("/about", "about", "company"),
              ("/about-us", "about-us", "company"), ("/team", "team", "founder"),
              ("/founders", "founders", "founder"), ("/product", "product", "company"),
              ("/careers", "careers", "hiring"), ("/blog", "blog", "activity"),
              ("/news", "news", "activity")]


def fetch_site(domain):
    """Homepage + about/team/product/careers/blog/news, fetched in parallel with hard wall-clock
    caps. Careers page doubles as a hiring signal; team/about pages give founder info."""
    out, seen_text = [], set()
    base = domain if domain.startswith("http") else f"https://{domain}"
    responses = _get_many([urljoin(base, path) for path, _, _ in SITE_PAGES], UA)
    for (path, label, cat), r in zip(SITE_PAGES, responses):
        if r is None or r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "noscript"]):
            t.decompose()
        title = (soup.title.string or "").strip() if soup.title and soup.title.string else label
        meta = soup.find("meta", attrs={"name": "description"})
        heads = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])][:12]
        paras = [x.get_text(" ", strip=True) for x in soup.find_all("p")
                 if len(x.get_text(strip=True)) > 40][:6]
        links = [a.get_text(" ", strip=True) for a in soup.find_all("a")
                 if 8 < len(a.get_text(strip=True)) < 90][:40] if label in ("careers", "blog", "news") else []
        text = " | ".join(filter(None, [meta["content"] if meta and meta.get("content") else "",
                                        *heads, *paras, *links]))[:2000]
        key = text[:300]
        if text and key not in seen_text:
            seen_text.add(key)
            out.append({"kind": f"site:{label}", "category": cat, "title": f"{title} ({label} page)",
                        "text": text, "url": r.url, "date": None})
    return out


def gather(p, id_prefix="E", employment=True, cap=40):
    """Collect evidence in separate research buckets (category field):
    person_news, employment, company_news, company, founder, activity, hiring, user_notes."""
    stage("Gather")
    ev = []
    print("  gathering: company news ...")
    ev += fetch_news(f'"{p["company"]}"', limit=10, category="company_news")
    print("  gathering: person news ...")
    ev += fetch_news(f'"{p["name"]}" "{p["company"]}"', limit=6, category="person_news")
    if employment:
        print("  gathering: employment / role-change signals ...")
        ev += fetch_news(f'"{p["name"]}" (joins OR appointed OR "steps down" OR "stepped down" '
                         f'OR "new role" OR "moves to" OR leaves OR former)',
                         limit=6, category="employment")
    if p.get("domain"):
        print("  gathering: company site ...")
        ev += fetch_site(p["domain"])
    if p.get("notes"):   # pasted LinkedIn posts, CRM notes, etc. (we don't scrape LinkedIn: ToS)
        ev.append({"kind": "user_notes", "category": "user_notes",
                   "title": "Rep-provided notes / LinkedIn text",
                   "text": p["notes"][:2500], "url": None, "date": None})
    seen, uniq = set(), []
    for e in ev:
        k = e["title"].lower()
        if k not in seen:
            seen.add(k); uniq.append(e)
    uniq = uniq[:cap]
    for i, e in enumerate(uniq, 1):
        e["id"] = f"{id_prefix}{i}"
    return uniq


CATEGORY_ORDER = [("user_notes", "REP-PROVIDED NOTES"),
                  ("employment", "EMPLOYMENT / ROLE-CHANGE SEARCH (may be about a different person with the same name - check)"),
                  ("person_news", "PERSON NEWS"),
                  ("founder", "FOUNDER / TEAM PAGES"),
                  ("company", "COMPANY & PRODUCT PAGES"),
                  ("company_news", "COMPANY NEWS"),
                  ("activity", "COMPANY ACTIVITY (blog / news pages)"),
                  ("hiring", "HIRING / CAREERS"),
                  ("successor", "SUCCESSOR SEARCH")]


def ev_text(e, n=None):
    if n is None:
        if e["kind"] == "user_notes":
            n = 1500
        elif e["kind"].startswith("site"):
            n = 1000
        else:
            n = 400
    return e["text"][:n]


def _ev_line(e):
    return f"[{e['id']}] kind={e['kind']} date={e['date'] or 'undated'} :: {e['title']} :: {ev_text(e)}"


def render_evidence(ev):
    lines, known = [], set()
    for cat, label in CATEGORY_ORDER:
        known.add(cat)
        grp = [e for e in ev if e.get("category") == cat]
        if grp:
            lines.append(f"### {label}")
            lines += [_ev_line(e) for e in grp]
    rest = [e for e in ev if e.get("category") not in known]
    if rest:
        lines.append("### OTHER")
        lines += [_ev_line(e) for e in rest]
    return "\n".join(lines)


# --------------------------------------------------------------------------- 2. ANALYSE
ANALYST_SYS = ("You are a rigorous B2B sales researcher. Use ONLY the evidence provided. "
               "Never invent facts, dates or quotes. If evidence is thin, say so. "
               "Reply with JSON only.")


def analyse(p, seller, ev):
    stage("Analyse")
    purpose = get_purpose(seller)
    prompt = f"""TODAY: {datetime.now().date().isoformat()}

*** PURPOSE OF THIS EMAIL (most important; drives which signals and facts matter): {purpose} ***

PROSPECT: {p['name']}, {p.get('title') or 'title unknown'} at {p['company']} (domain: {p.get('domain') or 'unknown'})
SENDER BACKGROUND (only relevant if the purpose involves our product): {seller['what_we_sell']}
PAINS WE SOLVE (only use these if the purpose is about selling / demoing / pitching our product): {', '.join(seller['typical_pains'])}

EVIDENCE (grouped by research bucket):
{render_evidence(ev)}

Tasks:
1. IDENTITY: Does the evidence plausibly concern THIS company (match domain / industry / people), not a
   different company with a similar name? Does any evidence about the person match this name+company?
2. EMPLOYMENT STATUS, exactly one of:
   "current"       - evidence supports that the person is still in this role/company (e.g. listed on the
                     company's own pages, recently quoted in that role).
   "moved"         - ONLY if evidence explicitly says THIS person left {p['company']} or took a new role
                     elsewhere (make sure it is the same person, not a namesake).
   "wrong_company" - the evidence clearly shows this person is known as founder/CEO/employee of a DIFFERENT
                     company, and nothing shows them at {p['company']}. Articles that merely compare or
                     mention both companies (competitor coverage, market comparisons) are NOT employment
                     evidence. The input name/company pair is probably a mistake.
   "unknown"       - otherwise.
   If "moved" or "wrong_company", fill new_position ONLY from evidence (the company they actually work at,
   and title if stated). Never guess. Otherwise set every new_position field to null.
3. FACTS: stable background facts about the company and founder that the evidence DIRECTLY states (max 8):
   what the company does, its product, target customers, industry, the person's role/founder status,
   publicly stated activity, hiring. No inference, no guesses. Each fact = one plain factual sentence.
   Rate each fact's "relevance" (0-10) by how useful it is for the PURPOSE above.
   Allowed categories: {', '.join(FACT_CATEGORIES)}.
4. SIGNALS: list candidate outreach hooks based on NEWS/recent events (max 6). Judge each signal by how naturally it
   supports the PURPOSE above ("{purpose}"). Different purposes should favour DIFFERENT signals (e.g. a partnership
   purpose favours launches/integrations/expansion; a sales purpose favours growth/hiring pains). Prefer signals that are
   specific to the company or person and recent. "relevance" (0-10) = fit with the PURPOSE. Mark sensitivity: "none" for
   neutral/positive news; "negative" for layoffs, lawsuits, breaches, outages, regulatory action, bad earnings,
   leadership exits; "grave" for deaths, tragedies, or anything a cold email would be tasteless about.
   An empty signals list is fine if there is no real news.

JSON schema:
{{"identity_confidence": 0.0-1.0, "identity_reasoning": "...",
  "prospect_status": "current|moved|wrong_company|unknown", "status_reasoning": "...",
  "new_position": {{"new_company": null, "new_title": null, "confidence": 0.0-1.0,
                   "summary": "one factual sentence about the move / the company they actually belong to", "evidence_ids": ["E1"]}},
  "facts": [{{"category": "one of the allowed categories", "fact": "one factual sentence",
             "evidence_ids": ["E1"], "relevance": 0-10}}],
  "signals": [{{"evidence_ids": ["E1"], "type": "funding|hire|launch|job_posting|exec_statement|expansion|challenge|negative|other",
               "summary": "one factual sentence", "why_it_matters_to_this_person": "how it connects to the PURPOSE",
               "relevance": 0-10, "sensitivity": "none|negative|grave"}}]}}"""
    return gemini(ANALYST_SYS, prompt, temperature=0.1)


def find_successor_for(p, ev, ev_by_id):
    """Optional: who appears to have taken over the prospect's old role? Recommendation only."""
    title = (p.get("title") or "").strip()
    q = '"' + p["company"] + '" '
    if title:
        q += '"' + title + '" '
    q += '(appoints OR names OR succeeds OR replaces OR "takes over" OR "new")'
    print("  searching for successor ...")
    ev_s = fetch_news(q, limit=6, category="successor")
    seen = {e["title"].lower() for e in ev}
    ev_s = [e for e in ev_s if e["title"].lower() not in seen]
    for i, e in enumerate(ev_s, 1):
        e["id"] = f"S{i}"
    allev = ev + ev_s
    if not allev:
        return None, []
    prompt = f"""TODAY: {datetime.now().date().isoformat()}
{p['name']} ({title or 'title unknown'}) appears to have LEFT {p['company']}.
Identify the person who appears to have TAKEN OVER that role at {p['company']}. Use ONLY the evidence.
Only name someone if the evidence explicitly indicates they replaced / succeeded / were appointed to that role.
Never name {p['name']} themself. If unclear, return null.

EVIDENCE:
{render_evidence(allev)}

JSON: {{"successor_name": null, "successor_title": null, "confidence": 0.0-1.0,
        "reasoning": "...", "evidence_ids": ["S1"]}}"""
    r = gemini(ANALYST_SYS, prompt, temperature=0.0)
    ids = [i for i in (r.get("evidence_ids") or []) if i in {e["id"] for e in allev}]
    name = (r.get("successor_name") or "").strip()
    if not name or not ids or _f(r.get("confidence")) < 0.6 or name.lower() == p["name"].lower():
        return None, ev_s
    return {"name": name, "title": r.get("successor_title"), "confidence": _f(r.get("confidence")),
            "reasoning": r.get("reasoning"), "evidence_ids": ids}, ev_s


# --------------------------------------------------------------------------- 3. JUDGE
def age_days(ev_by_id, ids):
    dates = [ev_by_id[i]["date"] for i in ids if i in ev_by_id and ev_by_id[i]["date"]]
    if not dates:
        return None
    newest = max(datetime.fromisoformat(d) for d in dates)
    return (datetime.now() - newest).days


def freshness(days):
    if days is None: return 0.5
    if days <= 30: return 1.0
    if days <= 90: return 0.8
    if days <= 180: return 0.5
    if days <= MAX_STALE_DAYS: return 0.2
    return 0.0


def score_signals(signals, ev_by_id):
    scored = []
    for s in signals or []:
        ids = [i for i in s.get("evidence_ids", []) if i in ev_by_id]
        if not ids:                      # hallucinated citation (or filtered as not-about-this-company) -> discard
            continue
        d = age_days(ev_by_id, ids)
        if d is not None and d > MAX_STALE_DAYS:
            continue
        fr = freshness(d)
        trust = 1.0 if any(ev_by_id[i]["kind"] == "user_notes" for i in ids) else 0.0
        s["age_days"], s["evidence_ids"] = d, ids
        s["score"] = round(min(10, _f(s.get("relevance", 0)) * 0.7 + fr * 10 * 0.3 + trust), 2)
        s.setdefault("sensitivity", "none")
        s.setdefault("why_it_matters_to_this_person", "")
        s.setdefault("type", "other")
        scored.append(s)
    return sorted(scored, key=lambda x: -x["score"])


def score_facts(facts, ev_by_id):
    """Keep only facts that cite real evidence ids (hallucinated citations are dropped)."""
    out = []
    for f in facts or []:
        if not isinstance(f, dict) or not f.get("fact"):
            continue
        ids = [i for i in (f.get("evidence_ids") or []) if i in ev_by_id]
        if not ids:
            continue
        f["evidence_ids"] = ids
        f["relevance"] = _f(f.get("relevance"), 5)
        f["category"] = f.get("category") or "other"
        out.append(f)
    return sorted(out, key=lambda x: -x["relevance"])


def build_company_signal(facts):
    """Bundle the best stable facts into a pseudo-signal so drafting/verifying can reuse the hook path."""
    top = facts[:4]
    ids = []
    for f in top:
        for i in f["evidence_ids"]:
            if i not in ids:
                ids.append(i)
    return {"type": "company_facts", "summary": " ; ".join(f["fact"] for f in top),
            "why_it_matters_to_this_person": "Stable public background about the company/founder (not news).",
            "evidence_ids": ids, "sensitivity": "none", "score": None, "age_days": None,
            "relevance": max(f["relevance"] for f in top)}


def select_approach(packet, a, ev_by_id, p):
    """Choose personalisation level 3 / 2 / 1 for a verified prospect.
    Returns (mode, signal, sensitive_ctx) or None if the packet was put on HOLD.
    Negative signals/facts are IGNORED (dropped); only grave events (death/tragedy) HOLD."""
    stage("Judge")
    # only keep evidence that is really about THIS company (or person news / own site / notes)
    ok = {i for i, e in ev_by_id.items()
          if mentions_company(e, p) or e.get("category") == "person_news"}
    for s in a.get("signals") or []:
        s["evidence_ids"] = [i for i in (s.get("evidence_ids") or []) if i in ok]
    for f in a.get("facts") or []:
        if isinstance(f, dict):
            f["evidence_ids"] = [i for i in (f.get("evidence_ids") or []) if i in ok]

    scored = score_signals(a.get("signals", []), ev_by_id)
    grave = [s for s in scored if s.get("sensitivity") == "grave"]
    negative = [s for s in scored if s.get("sensitivity") == "negative"]
    clean = [s for s in scored if s.get("sensitivity", "none") == "none"]
    packet["alternatives"] = clean[:4]
    # facts: drop anything that reads as negative so it can never reach the email
    packet["facts"] = [f for f in score_facts(a.get("facts"), ev_by_id)
                       if not _has_term(f["fact"], NEGATIVE_TERMS + GRAVE_TERMS)]

    if grave:                                         # death / tragedy -> still no draft
        top = grave[0]
        packet["status"] = "HOLD"
        packet["level"] = 0
        packet["flags"].append(f"GRAVE EVENT detected ({top['type']}): {top['summary']}. No email drafted. "
                               "Do not send now; re-evaluate later or use a human-led approach.")
        return None

    sctx = packet.get("negative_ctx")
    if negative:                                      # negative -> ignore it, write with the positive ones
        sctx = sctx or negative[0]["summary"]
        packet["flags"].append(f"NEGATIVE SIGNAL IGNORED ({negative[0]['type']}): {negative[0]['summary']}. "
                               "The draft uses positive/neutral facts only and does not mention it.")

    best = clean[0] if clean and clean[0]["score"] >= SCORE_THRESHOLD else None
    if best:                                                           # LEVEL 3
        return "hook", best, sctx
    if any(f.get("category") in FACT_CORE for f in packet["facts"]):   # LEVEL 2
        return "company", build_company_signal(packet["facts"]), sctx
    return "fallback", None, sctx                                      # LEVEL 1


# --------------------------------------------------------------------------- 4/5. DRAFT + VERIFY
WRITER_SYS = ("You write first-touch B2B emails that a busy executive would actually read. "
              "You write like a thoughtful person typing a quick note to someone they respect, "
              "not like a marketing template. You are honest, specific and brief. "
              "The stated PURPOSE of the email is the top priority: the whole email must clearly serve it. "
              "Reply with JSON only.")


def first_name(full):
    return (full or "").strip().split(" ")[0] or "there"


def frame_body(body, p, seller):
    """Strip any greeting/sign-off the model added, then wrap with ours:
    'Hi <first name>,' at the top and 'Thanks, <sender>' at the bottom."""
    body = (body or "").strip()
    body = re.sub(r"^\s*(?:hi|hello|hey|dear)\b[^\n]{0,30}\n+", "", body, flags=re.I)
    body = re.sub(r"\n+\s*(?:thanks|thank you|best|regards|cheers|sincerely)\b[\s\S]{0,60}$", "",
                  body, flags=re.I).strip()
    return f"Hi {first_name(p['name'])},\n\n{body}\n\nThanks,\n{seller['sender_name']}"


def _cited(ids, ev_by_id, n=1200):
    return "\n".join(f"- {ev_by_id[i]['title']} ({ev_by_id[i]['date'] or 'undated'}): {ev_text(ev_by_id[i], n)}"
                     for i in ids if i in ev_by_id)


def write_draft(p, seller, mode, signal, ev_by_id, feedback=None, sensitive_context=None, move=None):
    purpose = get_purpose(seller)
    if mode == "hook":
        brief = (f"HOOK (the ONLY specific fact you may reference):\n{signal['summary']}\n"
                 f"Evidence:\n{_cited(signal['evidence_ids'], ev_by_id)}\n"
                 f"Why it matters to them: {signal['why_it_matters_to_this_person']}")
    elif mode == "company":
        brief = ("COMPANY/FOUNDER FACTS (the ONLY specific facts you may reference; use one or two). These are "
                 "stable background facts, NOT news: never say 'recently', 'just', 'new' or imply a timeline. "
                 "Do not claim to have seen posts, news or activity.\n"
                 f"{signal['summary']}\nEvidence:\n{_cited(signal['evidence_ids'], ev_by_id)}")
    else:
        brief = ("LOW-INFORMATION PROSPECT (e.g. a very small / early-stage company with almost no public footprint). "
                 "Write a SHORT, GENERAL, warm and positive note. You may use ONLY: the person's name, their job "
                 "title (if given) and the company name exactly as provided. You may include one brief, generic, "
                 "positive remark about building a company / their role (e.g. that building something early-stage "
                 "takes real drive), but it must NOT contain any specific claim. Do NOT say what the company does, "
                 "its product, customers, traction, size, funding, location, news, posts or achievements. "
                 "Do NOT pretend you researched them or followed their work. Keep it light, then connect "
                 "straight to the PURPOSE of the email and ask for the one thing the purpose needs. "
                 "Do not go deep and do not assume any problems they have.")
    if move:
        brief += ("\n\nROLE CHANGE (supported by evidence; you may acknowledge it in ONE short clause, e.g. mention "
                  "the new role. Do not over-congratulate, do not speculate why they moved, do not mention "
                  f"anything about their old company beyond this):\n{move['summary']}\n"
                  f"Evidence:\n{_cited(move['evidence_ids'], ev_by_id)}")
    extra = ""
    if sensitive_context:
        extra += ("\nNOTE: some negative press exists about this company/person. Do NOT mention, allude to or "
                  "hint at any controversy, trouble, challenge or hardship. Write a warm, positive note "
                  "using only the facts in the brief above.")
    if feedback:
        extra += f"\nPREVIOUS DRAFT WAS REJECTED. Fix this: {feedback}"
    body_max = FALLBACK_BODY_WORDS if mode == "fallback" else MAX_WORDS - 15
    prompt = f"""Write a cold email from {seller['sender_name']} at {seller['company']} to {p['name']}, {p.get('title') or ''} at {p['company']}.

*** PURPOSE OF THIS EMAIL (TOP PRIORITY): {purpose} ***
The subject, the opening, the middle and the closing ask must ALL clearly serve this purpose. If the purpose is
NOT about selling (e.g. partnership, hiring, feedback, introduction, podcast, investment, research), do NOT pitch
the product and do NOT talk about the prospect's "pain points".
REQUESTED TONE: {seller.get('tone') or DEFAULT_SELLER['tone']}

Background on the sender (context only; mention the product/company offering ONLY if the purpose calls for it): {seller['what_we_sell']}

{brief}{extra}

RULES
- Body max {body_max} words, plain text, no emojis, no bullet points, no links.
- Do NOT write a greeting ("Hi ...") or a sign-off ("Thanks", "Best"). Those are added automatically. Write only the middle.
- Sentence 1: the specific observation (hook/company mode) or a general, positive role-relevant remark (role-based mode).
- Sentence 2-3: connect the observation to the PURPOSE in a natural way and say clearly why you're writing.
  Only if the purpose is about selling/pitching our product may you mention a problem they might have, and then
  phrase it as a HYPOTHESIS ("often means...", "I'd guess..."), never assert their internal problems.
- Do not flatter. Do not explain that you researched them. No filler openers.
- Do not state any fact about them that is not in the brief/evidence above.
- The PURPOSE text is the ONLY source of facts about any past interaction with them (a meeting, event, call,
  intro). NEVER invent when or where it happened, or what was discussed. If the purpose mentions one, refer to it
  in one short generic clause using only the details written in the purpose, then move to the next step.
- Close with one line that asks for the thing the PURPOSE needs. Keep it low-pressure; style hint: {seller['cta_style']}
  (if the style hint conflicts with what the purpose requires, the purpose wins).
- Subject: under 7 words, specific, lowercase-friendly, no clickbait, and reflecting the purpose.

TONE (this matters)
- Sound like a real person writing to one other person. Use contractions (I'd, it's, you're) and everyday words.
- Mix short and medium sentences. It's fine to start a sentence with "And" or "But" once.
- Warm and curious, never salesy. Ask the closing question like you genuinely want to know the answer.
- No corporate phrases, no buzzwords, no exclamation marks.

FINAL CHECK: the subject and every sentence must serve this purpose: {purpose}

JSON: {{"subject": "...", "body": "..."}}"""
    d = gemini(WRITER_SYS, prompt, temperature=0.7)
    d["subject"] = (d.get("subject") or "").strip()
    d["body"] = frame_body(d.get("body"), p, seller)
    return d


def verify_draft(draft, p, mode, signal, ev_by_id, move=None, purpose=""):
    parts = []
    if mode in ("hook", "company") and signal:
        parts.append("\n".join(f"- {ev_by_id[i]['title']}: {ev_text(ev_by_id[i], 1500)}"
                               for i in signal["evidence_ids"] if i in ev_by_id))
    if move:
        parts.append(f"- ROLE CHANGE: {move['summary']}\n" +
                     "\n".join(f"- {ev_by_id[i]['title']}: {ev_text(ev_by_id[i], 1500)}"
                               for i in move["evidence_ids"] if i in ev_by_id))
    if mode == "fallback":
        parts.append("(low-information mode: the ONLY allowed facts are the prospect's name, job title and "
                     "company name as listed above. Generic, non-specific positive remarks about building a "
                     "company or about the role are fine. Flag ANY claim about what the company does, its product, "
                     "customers, traction, size, funding, location, news, posts or achievements.)")
    elif not parts:
        parts.append("(role-based mode: the draft may only reference the person's job title and generic "
                     "role-level challenges)")
    facts = "\n".join(parts)
    prompt = f"""You are a fact-checker for outbound emails.
PROSPECT: {p['name']}, {p.get('title') or 'title unknown'}, {p['company']}
ALLOWED FACTS:
{facts}

SENDER'S STATED PURPOSE (the sender's own words; any past interaction it mentions, such as a meeting or event,
may be referred to in general terms, but details it does NOT state, like what was discussed, when or where, are
invented and must be flagged): {purpose or '(none)'}

DRAFT:
Subject: {draft['subject']}
{draft['body']}

List every statement that asserts a fact about the prospect or their company. Flag any not supported by the allowed
facts (including invented numbers, dates, quotes, or claims that they 'recently' did something not in evidence).
Hypotheses phrased as guesses ("I'd guess", "often") are fine. The greeting and sign-off are fine.
Statements about the SENDER's own intent or purpose for writing are fine.
JSON: {{"all_supported": true|false, "unsupported": ["..."]}}"""
    return gemini("You are a strict fact-checker. JSON only.", prompt, temperature=0.0)


def lint(draft):
    issues = []
    n = len(draft["body"].split())
    if n > MAX_WORDS: issues.append(f"too long ({n} words)")
    low = (draft["subject"] + " " + draft["body"]).lower()
    issues += [f"cliche phrase: '{b}'" for b in BANNED if b in low]
    if "http" in low: issues.append("contains a link")
    return issues


def draft_and_verify(packet, p, seller, mode, signal, ev_by_id, sensitive_ctx, feedback=None, move=None):
    packet["sensitive_ctx"] = sensitive_ctx
    packet["effective_prospect"] = p
    packet["purpose"] = get_purpose(seller)
    packet["flags"] = [f for f in packet["flags"] if not f.startswith("Draft failed automated checks")]
    base = feedback
    d = v = None
    for attempt in range(2):
        stage("Draft")
        print(f"  drafting (attempt {attempt + 1}) ...")
        d = write_draft(p, seller, mode, signal, ev_by_id, feedback, sensitive_ctx, move=move)
        issues = lint(d) + (negative_leak(d) if sensitive_ctx else [])
        stage("Verify")
        print("  verifying claims ...")
        v = verify_draft(d, p, mode, signal, ev_by_id, move=move, purpose=packet["purpose"])
        if v.get("all_supported") and not issues:
            packet["draft"], packet["verification"] = d, v
            return packet
        feedback = "; ".join(filter(None, [base] + list(v.get("unsupported", [])) + issues))
        print(f"  rejected: {feedback}")
    packet["draft"] = d
    packet["verification"] = v
    packet["status"] = "NEEDS_REVIEW"
    packet["flags"].append(f"Draft failed automated checks twice: {feedback}. Edit before use.")
    return packet


# --------------------------------------------------------------------------- EXPLANATION
def build_explanation(packet):
    """Why did the system choose this level/approach? Shown to the salesperson."""
    an = packet.get("analysis") or {}
    lvl = packet.get("level")
    status = an.get("prospect_status") or "unknown"
    sig, mv = packet.get("chosen_signal"), packet.get("move")
    flags = packet.get("flags") or []
    ev_by_id = {e["id"]: e for e in packet.get("evidence", [])}

    if packet["status"] == "HOLD":
        reason = "Held for safety: " + (flags[-1] if flags else "sensitive event detected")
        action = "Do not send now. Re-evaluate in 3-4 weeks or use a human-led approach."
    elif lvl == 0:
        reason = flags[-1] if flags else "Identity or current employment could not be verified."
        action = ("Do not send. Add the company domain or paste LinkedIn/CRM notes and re-run, "
                  "or verify manually.")
    elif lvl == 3:
        reason = (f"Strong evidence-backed signal ({sig['type']}, score {sig['score']}, "
                  f"age {sig['age_days']} days): {sig['summary']}")
        action = "Review the draft and approve/edit."
    elif lvl == 2:
        reason = ("No strong recent signal, so the email uses stable public company/founder facts: "
                  + (sig["summary"] if sig else ""))
        action = "Review the draft (light personalisation, no 'recent' claims) and approve/edit."
    elif packet.get("low_info"):
        reason = ("Very little public information about this person/company (common for small or early-stage "
                  "startups), so identity / current role could not be verified. A short, generic, positive "
                  "note was drafted that makes no claims about the company.")
        action = ("Quick manual check (LinkedIn / company site) that they are still at the company, then "
                  "approve/edit. Adding the domain or pasting LinkedIn text and re-running will personalise it more.")
    else:
        reason = "Very little useful public information; generic role-based email."
        action = "Spend ~5 minutes on manual research, or move to a nurture sequence instead of sending."

    if mv and lvl and lvl > 0:
        reason = (f"Prospect appears to have moved {mv['old_company']} to {mv['new_company']}"
                  f"{' as ' + mv['new_title'] if mv.get('new_title') else ''}. " + reason)
        action = "Verify the move on LinkedIn/CRM first, then send the new-role draft. " + \
                 ("Also consider the successor at the old company." if packet.get("successor") else "")
    elif mv is None and status == "moved":
        action = "Look up the prospect's new role manually; " + action

    ic = packet.get("input_correction")
    if ic and lvl:
        reason = (f"Input looks wrong: {packet['prospect']['name']} is associated with {ic['new_company']}"
                  f"{' as ' + ic['new_title'] if ic.get('new_title') else ''}, not "
                  f"{ic['old_company']}. " + reason)
        action = "Confirm the correct company/person, then review the draft. "

    if packet.get("suppressed_negative") and lvl:
        action += (" Negative news was found and deliberately left out of the draft; consider whether "
                   "the timing is right to send.")

    ids = []
    for src in (sig, mv or ic):
        for i in (src or {}).get("evidence_ids", []):
            if i not in ids:
                ids.append(i)
    evidence = [{"id": i, "title": ev_by_id[i]["title"], "url": ev_by_id[i]["url"]}
                for i in ids if i in ev_by_id]
    return {"person_status": status.replace("_", " ").capitalize(), "personalization_level": lvl,
            "level_name": LEVEL_NAMES.get(lvl, "-"), "reason": reason, "evidence": evidence,
            "recommended_action": action}


def finish(packet):
    packet["explanation"] = build_explanation(packet)
    return packet


# --------------------------------------------------------------------------- ORCHESTRATE
def block(packet, msg):
    """LEVEL 0: prospect moved and the new role can't be found, or wrong input that can't be corrected."""
    packet["status"] = "BLOCKED"
    packet["level"] = 0
    packet["mode"] = None
    packet["flags"].append(msg)
    return finish(packet)


def soft_fallback(packet, p, seller, ev_by_id, why):
    """LOW-INFO PROSPECT (tiny founder / early-stage startup with almost no public footprint).
    Instead of blocking, write a short, generic, positive Level-1 note that uses ONLY the name, title
    and company name the rep supplied (no evidence, no company claims). Always flagged for review."""
    packet["low_info"] = True
    packet["flags"].append(f"LOW-INFO PROSPECT: {why} A short, generic, positive role-based note was drafted "
                           "with no claims about the company. Verify the person is still at the company "
                           "before sending.")
    return finalize(packet, p, seller, ev_by_id, "fallback", None, None)


def finalize(packet, p, seller, ev_by_id, mode, signal, sensitive_ctx, move=None):
    packet["mode"], packet["chosen_signal"] = mode, signal
    packet["level"] = {"hook": 3, "company": 2, "fallback": 1}[mode]
    if signal:
        packet["tried_hooks"].append(signal["summary"])
    if mode == "company":
        packet["flags"].append("LIGHT PERSONALISATION: no strong recent signal; email uses stable public "
                               "company/founder facts only (no 'recent' claims).")
    elif mode == "fallback":
        packet["status"] = "NEEDS_REVIEW"
        packet["flags"].append("LOW PERSONALISATION (no usable company/founder facts or news). Draft is "
                               "role-based only. Recommend: 5 min of manual research, or move to a nurture "
                               "sequence instead of sending.")
    if move:
        packet["status"] = "NEEDS_REVIEW"
        packet["flags"].append("Draft targets the NEW role/company. Verify the move before sending.")
    draft_and_verify(packet, p, seller, mode, signal, ev_by_id, sensitive_ctx, move=move)
    return finish(packet)


def _actions_for_move(packet, p, move, successor):
    acts = []
    if move:
        acts.append(f"Follow {p['name']} to {move['new_company']}"
                    f"{' (' + move['new_title'] + ')' if move.get('new_title') else ''} - draft below.")
    else:
        acts.append(f"Look up {p['name']}'s new company/role manually (not reliably found in public evidence).")
    if successor:
        acts.append(f"Consider contacting the successor at {p['company']}: {successor['name']}"
                    f"{' (' + successor['title'] + ')' if successor.get('title') else ''}.")
    else:
        acts.append(f"Successor at {p['company']} not identified from public evidence; check their site/LinkedIn.")
    packet["actions"] = acts


def handle_moved(packet, p, seller, a, ev, ev_by_id, find_succ, mismatch=False):
    """mismatch=False: the prospect moved jobs. mismatch=True: the name/company input is wrong
    (e.g. Deepinder Goel + Swiggy) -> find the company they really belong to and draft for that.
    All recommendations, nothing automatic."""
    if mismatch:
        packet["flags"].append(f"INPUT MISMATCH: evidence ties {p['name']} to a different company than "
                               f"'{p['company']}': {a.get('status_reasoning')}")
    else:
        packet["flags"].append(f"PROSPECT MAY HAVE LEFT / CHANGED ROLE: {a.get('status_reasoning')}")

    successor = None
    if find_succ and not mismatch:
        successor, ev_s = find_successor_for(p, ev, ev_by_id)
        for e in ev_s:
            ev_by_id[e["id"]] = e
        packet["evidence"] = ev + ev_s
        packet["successor"] = successor

    npos = a.get("new_position") or {}
    new_company = (npos.get("new_company") or "").strip()
    nids = [i for i in (npos.get("evidence_ids") or []) if i in ev_by_id]
    if not (new_company and nids and npos.get("summary") and _f(npos.get("confidence")) >= 0.6):
        if mismatch:
            packet["actions"] = [f"Check your input: {p['name']} is not confirmed at {p['company']} and the "
                                 "real company could not be identified. Fix the name/company and re-run."]
            return block(packet, "Name and company do not appear to belong together and the correct company "
                                 "could not be identified. Verify the input manually.")
        _actions_for_move(packet, p, None, successor)
        return block(packet, "Prospect appears to have moved, but the new company/role could not be "
                             "reliably identified from public evidence. Verify manually.")

    move = {"old_company": p["company"], "old_title": p.get("title"), "new_company": new_company,
            "new_title": (npos.get("new_title") or "").strip() or None,
            "confidence": _f(npos.get("confidence")), "summary": npos["summary"], "evidence_ids": nids}

    if mismatch:
        packet["input_correction"] = move
        packet["status"] = "NEEDS_REVIEW"
        packet["flags"].append(f"Drafting for {new_company}{' as ' + move['new_title'] if move['new_title'] else ''}, "
                               f"the company the evidence points to, NOT '{p['company']}'. "
                               "Confirm this is the person you meant before sending.")
        packet["actions"] = [f"Correct the company in your list: {p['name']} -> {new_company}.",
                             "Review the draft below (written for the corrected company)."]
        move_for_draft = None            # it is not a job change, so the email must not mention one
    else:
        packet["move"] = move
        packet["flags"].append(f"PROSPECT MOVED (per evidence): {p['company']} to {new_company}"
                               f"{' as ' + move['new_title'] if move['new_title'] else ''}. "
                               "Verify on LinkedIn/CRM before sending.")
        _actions_for_move(packet, p, move, successor)
        move_for_draft = move

    p2 = {"name": p["name"], "company": new_company, "title": move["new_title"],
          "domain": p.get("new_domain") or None, "notes": None, "email": p.get("email")}
    print(f"  researching the {'corrected' if mismatch else 'new'} company: {new_company}")
    ev2 = gather(p2, id_prefix="N", employment=False)
    ev2, grave = suppress_negative(packet, p2, ev2)
    if grave:
        return finish(packet)
    for e in ev2:
        ev_by_id[e["id"]] = e
    packet["evidence"] = packet["evidence"] + ev2
    print("  analysing company ...")
    a2 = analyse(p2, seller, [ev_by_id[i] for i in nids] + ev2)
    packet["analysis_new_company"] = {k: a2.get(k) for k in ("identity_confidence", "identity_reasoning")}
    if _f(a2.get("identity_confidence")) < 0.5:
        return block(packet, f"Could not confirm the company/role ({a2.get('identity_reasoning')}). "
                             "Verify manually or add the company domain.")

    sel = select_approach(packet, a2, ev_by_id, p2)
    if sel is None:
        return finish(packet)
    mode, signal, sctx = sel
    return finalize(packet, p2, seller, ev_by_id, mode, signal, sctx, move=move_for_draft)


def process(p, seller, find_successor=True):
    packet = {"prospect": p, "effective_prospect": None, "status": "READY", "mode": None, "level": None,
              "flags": [], "chosen_signal": None, "alternatives": [], "facts": [], "draft": None,
              "analysis": None, "move": None, "successor": None, "actions": [], "explanation": None,
              "attempts": 1, "tried_hooks": [], "sensitive_ctx": None, "evidence": [],
              "purpose": get_purpose(seller), "low_info": False, "input_correction": None,
              "suppressed_negative": [], "negative_ctx": None}
    print(f"\n> {p['name']} - {p['company']}")
    print(f"  purpose: {packet['purpose']}")
    ev = gather(p)
    ev, grave = suppress_negative(packet, p, ev)     # negative news is dropped; only deaths/tragedies HOLD
    if grave:
        return finish(packet)
    ev_by_id = {e["id"]: e for e in ev}
    packet["evidence"] = ev

    # SMALL STARTUP: nothing public found at all (or only negative news, which we ignore)
    if not ev:
        return soft_fallback(packet, p, seller, ev_by_id,
                             "No public evidence found (very small or new company / founder).")

    # SMALL STARTUP / NAME CLASH: results exist but none actually mention this company
    # (employment search is person-only, so it is excluded from this check)
    if not any(mentions_company(e, p) for e in ev if e.get("category") != "employment"):
        return soft_fallback(packet, p, seller, ev_by_id,
                             "Nothing found actually mentions this company (very small company or a "
                             "name clash with other companies).")

    print("  analysing evidence ...")
    a = analyse(p, seller, ev)
    status = a.get("prospect_status")
    if status not in ("current", "moved", "wrong_company", "unknown"):
        status = "unknown"
    conf = _f(a.get("identity_confidence"))
    packet["analysis"] = {"identity_confidence": a.get("identity_confidence"),
                          "identity_reasoning": a.get("identity_reasoning"),
                          "prospect_status": status, "status_reasoning": a.get("status_reasoning")}

    # WRONG INPUT (e.g. Deepinder Goel + Swiggy): checked BEFORE the identity fallback, because a wrong
    # pair usually also produces a low identity score and would otherwise get a generic mail for the
    # wrong company.
    if status == "wrong_company":
        return handle_moved(packet, p, seller, a, ev, ev_by_id, False, mismatch=True)

    # LOW-INFO: identity can't be confirmed. The fallback uses no evidence, so a namesake can't leak in.
    if conf < 0.5:
        return soft_fallback(packet, p, seller, ev_by_id,
                             f"Identity could not be confirmed ({a.get('identity_confidence')}): "
                             f"{a.get('identity_reasoning')}.")

    # PERSON MOVED (e.g. Kunal Shah): new company + role, draft for the new role
    if status == "moved":
        return handle_moved(packet, p, seller, a, ev, ev_by_id, find_successor)

    # LOW-INFO: current employment not verified AND identity only weakly confirmed
    if status == "unknown" and conf < 0.65:
        return soft_fallback(packet, p, seller, ev_by_id,
                             f"Current employment not verified and identity only weakly confirmed ({conf}): "
                             f"{a.get('status_reasoning')}.")
    if status == "unknown":
        packet["flags"].append("Current employment not explicitly confirmed by evidence; verify in CRM/LinkedIn.")

    sel = select_approach(packet, a, ev_by_id, p)
    if sel is None:
        return finish(packet)       # HOLD
    mode, signal, sctx = sel
    return finalize(packet, p, seller, ev_by_id, mode, signal, sctx)


def regenerate(packet, seller):
    """Rep rejected the current draft -> write a NEW one.
    Prefers the next-best unused news hook; if none is left, writes a fresh angle on the same
    hook / facts (or a fresh role-based note). Same verification + lint as the first draft."""
    p = packet.get("effective_prospect") or packet["prospect"]
    ev_by_id = {e["id"]: e for e in packet["evidence"]}
    tried = packet.setdefault("tried_hooks", [])
    packet["attempts"] = packet.get("attempts", 1) + 1
    old = packet.get("draft") or {}

    # If the purpose changed since the research/analysis ran, the hooks were ranked for the OLD purpose.
    new_purpose = get_purpose(seller)
    purpose_changed = bool(packet.get("purpose")) and packet["purpose"] != new_purpose
    print(f"  regenerate: old purpose = {packet.get('purpose')!r} | new purpose = {new_purpose!r} "
          f"| changed = {purpose_changed}")
    packet["flags"] = [f for f in packet["flags"] if not f.startswith("PURPOSE CHANGED")]
    if purpose_changed:
        packet["flags"].append("PURPOSE CHANGED since the research was done: the new draft follows the new purpose, "
                               "but the hooks/facts were ranked for the old one. Re-run the prospect from scratch "
                               "for the best match.")

    nxt = None
    if not purpose_changed:
        nxt = next((s for s in packet.get("alternatives", [])
                    if s.get("sensitivity", "none") == "none"
                    and s["score"] >= SCORE_THRESHOLD
                    and s["summary"] not in tried), None)
    if nxt:
        print(f"  regenerating with next-best hook: {nxt['summary'][:80]}")
        mode, signal = "hook", nxt
        tried.append(nxt["summary"])
        packet["mode"], packet["chosen_signal"], packet["level"] = "hook", nxt, 3
        packet["flags"] = [f for f in packet["flags"]
                           if not f.startswith(("LOW PERSONALISATION", "LIGHT PERSONALISATION"))]
    else:
        print("  no unused hook left (or purpose changed); writing a fresh angle on the same one")
        mode, signal = packet.get("mode") or "fallback", packet.get("chosen_signal")

    if purpose_changed:
        # don't anchor on the old draft, and don't leave the OLD purpose's reasoning in the hook.
        feedback = (f"The PURPOSE of this email has CHANGED. The NEW purpose is: {new_purpose}. "
                    "Write a completely new email from scratch for this new purpose. Do NOT reuse the subject, "
                    "angle, wording, pitch or closing ask of any earlier draft.")
        if signal:
            signal = dict(signal)   # copy so we don't mutate the stored alternatives
            signal["why_it_matters_to_this_person"] = (
                f"Connect this fact to the NEW purpose yourself: {new_purpose}. "
                "Ignore any earlier framing of why it matters.")
            packet["chosen_signal"] = signal
    else:
        feedback = ("The rep rejected the previous draft. Write a clearly DIFFERENT email: a new opening line, "
                    "a different angle, and a different closing question. "
                    f"Previous subject: '{old.get('subject', '')}'. Previous body: '{old.get('body', '')[:300]}'. "
                    "Do not reuse its wording.")
    draft_and_verify(packet, p, seller, mode, signal, ev_by_id, packet.get("sensitive_ctx"),
                     feedback=feedback, move=packet.get("move"))
    return finish(packet)


# --------------------------------------------------------------------------- 6. HUMAN REVIEW
def slug(s): return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def show(packet):
    p = packet["prospect"]
    bar = "=" * 72
    print(f"\n{bar}\n{p['name']} | {p.get('title') or ''} | {p['company']}   "
          f"[{packet['status']}] mode={packet['mode']} version={packet.get('attempts', 1)}")
    if packet.get("purpose"):
        print(f"  PURPOSE: {packet['purpose']}")
    ex = packet.get("explanation")
    if ex:
        print(f"  PERSON STATUS: {ex['person_status']}   |   LEVEL: {ex['personalization_level']} - {ex['level_name']}")
        print(f"  WHY: {ex['reason']}")
        print(f"  NEXT: {ex['recommended_action']}")
    for f in packet["flags"]:
        print(f"  ! {f}")
    mv = packet.get("move")
    if mv:
        print(f"\n  MOVE: {mv['old_company']} -> {mv['new_company']}"
              f"{' (' + mv['new_title'] + ')' if mv.get('new_title') else ''}  conf={mv['confidence']}")
    ic = packet.get("input_correction")
    if ic:
        print(f"\n  INPUT CORRECTION: {packet['prospect']['company']} -> {ic['new_company']}"
              f"{' (' + ic['new_title'] + ')' if ic.get('new_title') else ''}  conf={ic['confidence']}")
    for act in packet.get("actions", []):
        print(f"  - {act}")
    s = packet.get("chosen_signal")
    if s:
        if s["type"] == "company_facts":
            print(f"\n  FACTS USED: {s['summary']}")
        else:
            print(f"\n  HOOK ({s['type']}, score {s['score']}, age {s['age_days']}d): {s['summary']}")
            print(f"  WHY: {s['why_it_matters_to_this_person']}")
        for i in s["evidence_ids"]:
            e = next((x for x in packet["evidence"] if x["id"] == i), None)
            if e:
                print(f"  SRC [{i}] {e['title'][:80]}  {e['url'] or ''}")
    if packet.get("draft"):
        print(f"\n  Subject: {packet['draft']['subject']}\n")
        for line in packet["draft"]["body"].splitlines():
            print("  " + line)
    print(bar)


def edit_text(text):
    editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write(text); path = f.name
    subprocess.call([editor, path])
    return Path(path).read_text().strip()


def review(packet, seller):
    show(packet)
    if not packet.get("draft"):
        return "skipped"
    while True:
        c = input("[a]pprove  [e]dit  [n]ew draft  [s]kip > ").strip().lower()
        if c == "e":
            packet["draft"]["body"] = edit_text(packet["draft"]["body"]); show(packet)
        elif c == "n":
            regenerate(packet, seller); show(packet)
        elif c == "a":
            if packet["status"] != "READY":
                if input(f"Status is {packet['status']}. Approve anyway? [y/N] ").lower() != "y":
                    continue
            return "approved"
        elif c == "s":
            return "skipped"


def save(packet, decision):
    OUT_DIR.mkdir(exist_ok=True)
    name = slug(f"{packet['prospect']['name']}-{packet['prospect']['company']}")
    (OUT_DIR / f"{name}.json").write_text(json.dumps({**packet, "decision": decision}, indent=2))
    if decision == "approved":
        d = packet["draft"]
        (OUT_DIR / f"{name}.APPROVED.txt").write_text(
            f"To: {packet['prospect'].get('email') or '(add email)'}\nSubject: {d['subject']}\n\n{d['body']}\n")
        print(f"  saved to {OUT_DIR}/{name}.APPROVED.txt (NOT sent - paste into your sequencer)")


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name"); ap.add_argument("--company"); ap.add_argument("--title")
    ap.add_argument("--domain", help="company website, e.g. acme.com (strongly improves accuracy)")
    ap.add_argument("--new-domain", help="website of the prospect's NEW company, if you already know they moved")
    ap.add_argument("--notes", help="pasted LinkedIn posts / CRM notes")
    ap.add_argument("--csv", help="batch file with columns: name,company,title,domain,new_domain,notes,email")
    ap.add_argument("--seller-file", help="JSON overriding the default seller profile")
    ap.add_argument("--sender-name", help="your name, used in the 'Thanks, <name>' sign-off")
    ap.add_argument("--purpose", help="why you are writing, e.g. 'explore a partnership'")
    ap.add_argument("--about", help="who you are / what you offer")
    ap.add_argument("--my-company", help="your company name")
    ap.add_argument("--tone", help="e.g. 'warm and casual'")
    ap.add_argument("--no-successor", action="store_true", help="skip successor search when a prospect has moved")
    ap.add_argument("--no-review", action="store_true", help="batch mode: save packets without prompting")
    a = ap.parse_args()

    seller = json.loads(Path(a.seller_file).read_text()) if a.seller_file else dict(DEFAULT_SELLER)
    if a.sender_name:
        seller["sender_name"] = a.sender_name
    for arg, k in (("purpose", "purpose"), ("tone", "tone"), ("about", "what_we_sell"), ("my_company", "company")):
        if getattr(a, arg):
            seller[k] = getattr(a, arg)
    if a.csv:
        prospects = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8")))
    elif a.name and a.company:
        prospects = [{"name": a.name, "company": a.company, "title": a.title,
                      "domain": a.domain, "new_domain": a.new_domain, "notes": a.notes}]
    else:
        ap.error("give --name and --company, or --csv")

    for p in prospects:
        try:
            packet = process(p, seller, find_successor=not a.no_successor)
        except Exception as e:
            print(f"  failed for {p.get('name')}: {e}"); continue
        if a.no_review:
            show(packet); save(packet, "pending_review")
        else:
            save(packet, review(packet, seller))


if __name__ == "__main__":
    main()