import contextlib, csv, html, io, os, uuid

import pandas as pd
import streamlit as st

import outreach as o   # the research pipeline (unchanged logic)

st.set_page_config(page_title="Outreach Research Agent", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1 { font-size: 1.75rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }
h4 { font-weight: 600 !important; }

.stage-row { display: flex; gap: 8px; margin: 4px 0 10px 0; }
.stage { flex: 1; padding: 9px 12px; border-radius: 6px; font-size: 0.85rem;
         border: 1px solid rgba(128,128,128,0.35); color: rgba(128,128,128,0.95); }
.stage.run  { border-color: #2563eb; color: #2563eb; background: rgba(37,99,235,0.08); font-weight: 600; }
.stage.done { border-color: rgba(22,163,74,0.6); color: #16a34a; background: rgba(22,163,74,0.07); }

.badge { display: inline-block; padding: 2px 10px; margin-left: 10px; border-radius: 999px;
         font-size: 0.78rem; font-weight: 600; vertical-align: middle; }
.badge.ok   { background: rgba(22,163,74,0.14);  color: #16a34a; }
.badge.warn { background: rgba(217,119,6,0.16);  color: #d97706; }
.badge.bad  { background: rgba(220,38,38,0.14);  color: #dc2626; }
.badge.muted{ background: rgba(128,128,128,0.18); color: gray; }
</style>
""", unsafe_allow_html=True)

st.title("Outreach research agent")
st.caption("Research a prospect, verify every claim against evidence, then review the draft. Nothing is ever sent.")

STAGES = ["Gather", "Analyse", "Judge", "Draft", "Verify"]
STATUS_LABEL = {"READY": "Ready", "NEEDS_REVIEW": "Needs review", "HOLD": "On hold", "BLOCKED": "Blocked"}
STATUS_CLASS = {"READY": "ok", "NEEDS_REVIEW": "warn", "HOLD": "bad", "BLOCKED": "bad"}

PURPOSES = ["Start a sales conversation / intro to our product",
            "Propose a partnership or collaboration",
            "Recruit / hiring outreach",
            "Investor or fundraising intro",
            "Ask for an introduction or advice",
            "Follow up after an event or meeting",
            "Other (describe below)"]
TONES = ["Friendly and professional", "Warm and casual", "Formal", "Short and direct"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Settings")
    key = st.text_input("Gemini API key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    model = st.text_input("Model", value=os.environ.get("GEMINI_MODEL", o.MODEL))
    o.API_KEY, o.MODEL = key, model
    find_succ = st.checkbox("Look for a successor if the prospect moved", value=True,
                            help="Uses one extra model call per moved prospect. Recommendation only.")
    st.divider()
    with st.expander("How it works"):
        st.markdown(
            "**Pipeline:** gather, analyse, judge, draft, verify, then human review.\n\n"
            "**Personalisation level**\n"
            "- 3, Personal: a recent, evidence-backed hook\n"
            "- 2, Company: stable public company or founder facts\n"
            "- 1, Role-based: very little public information\n"
            "- 0, Blocked: identity or employment not verified\n\n"
            "**Status**\n"
            "- Ready: passed all checks\n"
            "- Needs review: low personalisation, moved prospect, or failed checks\n"
            "- On hold: sensitive event, do not send\n"
            "- Blocked: cannot be verified")

# ------------------------------------------------------------------ sender details
with st.expander("Your details and purpose of the email", expanded=True):
    e1, e2 = st.columns(2)
    sender = e1.text_input("Your name", placeholder="Rishabh", help="Used in the sign-off.")
    my_company = e2.text_input("Your company", value="xyz")
    about = st.text_area("What you offer", value="fast delivery of fashion products", height=80)
    f1, f2 = st.columns(2)
    preset = f1.selectbox("Purpose", PURPOSES)
    tone = f2.selectbox("Tone", TONES)
    extra = st.text_input("Anything specific to mention or ask (optional)",
                          placeholder="Example: we met at SaaStr Bengaluru on 12 Sept",
                          help="For follow-ups, write where/when you met. The email can only use details you write here.")

purpose = "" if preset.startswith("Other") else preset
if extra.strip():
    purpose = f"{purpose}. {extra.strip()}" if purpose else extra.strip()
seller = {**o.DEFAULT_SELLER,
          "sender_name": sender.strip() or o.DEFAULT_SELLER["sender_name"],
          "company": my_company.strip() or o.DEFAULT_SELLER["company"],
          "what_we_sell": about.strip() or o.DEFAULT_SELLER["what_we_sell"],
          "purpose": purpose or o.DEFAULT_SELLER["purpose"],
          "tone": tone}


# ------------------------------------------------------------------ progress helpers
class Tracker:
    """Shows the five pipeline stages and which one is running."""
    def __init__(self, slot):
        self.slot, self.idx = slot, -1
        self.draw()

    def set(self, name):
        if name in STAGES:
            self.idx = STAGES.index(name)
            self.draw()

    def finish(self):
        self.idx = min(self.idx + 1, len(STAGES))   # last reached stage becomes done
        self.draw()

    def draw(self):
        cells = []
        for i, s in enumerate(STAGES):
            cls = "done" if i < self.idx else "run" if i == self.idx else ""
            cells.append(f'<div class="stage {cls}">{s}</div>')
        self.slot.markdown(f'<div class="stage-row">{"".join(cells)}</div>', unsafe_allow_html=True)


class LiveLog(io.StringIO):
    """Captures print() output from the pipeline and shows the latest line."""
    def __init__(self, slot):
        super().__init__()
        self.slot = slot

    def write(self, s):
        r = super().write(s)
        lines = [ln.strip() for ln in self.getvalue().splitlines() if ln.strip()]
        if lines:
            self.slot.text(lines[-1][:140])
        return r


def progress_ui():
    tracker = Tracker(st.empty())
    activity = st.empty()
    return tracker, activity


def _run(fn, tracker, activity):
    """Run a pipeline function with stdout captured and stage updates wired to the tracker."""
    log, err, out = LiveLog(activity), None, None
    o.STAGE_HOOK = tracker.set
    try:
        with contextlib.redirect_stdout(log):
            try:
                out = fn()
            except SystemExit as e:      # the pipeline calls sys.exit() on fatal API errors
                err = str(e)
            except Exception as e:
                err = str(e)
    finally:
        o.STAGE_HOOK = None
    tracker.finish()
    activity.empty()
    return out, err, log.getvalue()


def run_one(prospect, tracker, activity, seller, find_successor=True):
    packet, err, log = _run(lambda: o.process(prospect, seller, find_successor=find_successor),
                            tracker, activity)
    return {"packet": packet, "log": log, "error": err, "uid": uuid.uuid4().hex[:8]}


def regen_one(result, seller, tracker, activity):
    _, err, log = _run(lambda: o.regenerate(result["packet"], seller), tracker, activity)
    result["log"] += "\n--- new draft ---\n" + log
    return err


def _link(e):
    return f"[{e['title'][:90]}]({e['url']})" if e.get("url") else e["title"][:90]


def badge(status):
    return (f'<span class="badge {STATUS_CLASS.get(status, "muted")}">'
            f'{STATUS_LABEL.get(status, status)}</span>')


# ------------------------------------------------------------------ result view
def render(result, prefix, seller):
    if result["error"] or not result["packet"]:
        st.error(f"Run failed: {result['error']}")
        with st.expander("Run log"):
            st.code(result["log"], language=None)
        return

    pk = result["packet"]
    p = pk["prospect"]
    ex = pk.get("explanation") or {}
    version = pk.get("attempts", 1)
    kp = f"{prefix}_{result.get('uid', 'x')}"
    lvl = pk.get("level")
    lvl_txt = f"{lvl}, {ex.get('level_name', '')}" if lvl is not None else "-"
    v = pk.get("verification") or {}
    check = "Passed" if v.get("all_supported") else ("Flagged" if pk.get("draft") else "-")

    st.markdown(f"#### {html.escape(p['name'])}, {html.escape(p['company'])}{badge(pk['status'])}",
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Personalisation level", lvl_txt)
    m2.metric("Person status", ex.get("person_status") or "-")
    m3.metric("Fact check", check)
    m4.metric("Draft version", version)

    # CHANGED: tell the user when the purpose on screen differs from the one this draft was written for.
    # Changing the purpose does not rewrite the existing draft by itself; "New draft" does.
    if pk.get("draft") and pk.get("purpose") and pk["purpose"] != seller["purpose"]:
        st.warning("The purpose has changed since this draft was written. "
                   "Click **New draft** to rewrite it for the new purpose.\n\n"
                   f"- Draft was written for: {pk['purpose']}\n"
                   f"- Current purpose: {seller['purpose']}")
    elif pk.get("draft") and pk.get("purpose"):
        st.caption(f"Draft written for purpose: {pk['purpose']}")

    left, right = st.columns([3, 2], gap="large")

    # ---- assessment
    with right:
        st.markdown("**Assessment**")
        st.write(ex.get("reason") or "-")
        st.markdown("**Recommended action**")
        st.write(ex.get("recommended_action") or "-")
        if pk.get("actions"):
            for act in pk["actions"]:
                st.markdown(f"- {act}")
        if pk["flags"]:
            st.warning("**Notes**\n\n" + "\n".join(f"- {f}" for f in pk["flags"]))

    # ---- draft
    with left:
        d = pk.get("draft")
        if d:
            if not v.get("all_supported"):
                st.error("Fact check flagged: " + "; ".join(v.get("unsupported", ["unknown"])))
            subject = st.text_input("Subject", d["subject"], key=f"{kp}_subj_{version}")
            body = st.text_area("Body", d["body"], height=280, key=f"{kp}_body_{version}")
            st.caption(f"{len(body.split())} words (limit {o.MAX_WORDS})")

            b1, b2, b3, _ = st.columns([1, 1.4, 1, 3])
            if b1.button("Approve", key=f"{kp}_ok_{version}", type="primary"):
                pk["draft"] = {"subject": subject, "body": body}
                o.save(pk, "approved")
                st.success(f"Saved to out/{o.slug(p['name'] + '-' + p['company'])}.APPROVED.txt (not sent)")
            if b2.button("New draft", key=f"{kp}_regen_{version}"):
                tracker, activity = progress_ui()
                err = regen_one(result, seller, tracker, activity)
                if err:
                    st.error(f"Could not write a new draft: {err}")
                else:
                    st.rerun()
            if b3.button("Skip", key=f"{kp}_skip_{version}"):
                o.save(pk, "skipped")
                st.info("Skipped.")
        else:
            st.info("No draft was generated for this prospect. See the assessment for the reason.")

    # ---- detail tabs
    t_research, t_evidence, t_log, t_raw = st.tabs(["Research", "Evidence", "Run log", "Raw data"])

    with t_research:
        s = pk.get("chosen_signal")
        if s:
            ev = {e["id"]: e for e in pk["evidence"]}
            if s["type"] == "company_facts":
                st.markdown("**Company and founder facts used** (stable background, not news)")
                st.write(s["summary"])
            else:
                st.markdown(f"**Hook used** ({s['type']}, score {s['score']}, age {s['age_days']} days)")
                st.write(s["summary"])
                st.caption(f"Why it matters: {s['why_it_matters_to_this_person']}")
            for i in s["evidence_ids"]:
                if i in ev:
                    st.markdown(f"- Source [{i}]: {_link(ev[i])}")

        mv = pk.get("move")
        if mv:
            st.markdown("**Prospect moved**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Old company", mv["old_company"])
            c2.metric("New company", mv["new_company"])
            c3.metric("New role", mv.get("new_title") or "Unknown")
            st.caption(f"{mv['summary']} (confidence {mv['confidence']})")
        sc = pk.get("successor")
        if sc:
            st.markdown("**Possible successor at the old company**")
            st.write(f"{sc['name']}" + (f", {sc['title']}" if sc.get("title") else "")
                     + f" (confidence {sc['confidence']})")
            st.caption(sc.get("reasoning") or "")

        a = pk.get("analysis") or {}
        if a:
            st.markdown("**Analyst reasoning**")
            st.write(f"Identity ({a.get('identity_confidence')}): {a.get('identity_reasoning')}")
            st.write(f"Status ({a.get('prospect_status')}): {a.get('status_reasoning')}")
            if pk.get("analysis_new_company"):
                st.write("New company identity: " + str(pk["analysis_new_company"].get("identity_reasoning")))

        if pk.get("facts"):
            st.markdown(f"**Facts extracted ({len(pk['facts'])})**")
            fc = ["category", "fact", "relevance", "evidence_ids"]
            st.dataframe(pd.DataFrame([{c: x.get(c) for c in fc} for x in pk["facts"]]))
        if pk["alternatives"]:
            st.markdown("**News signals considered**")
            cols = ["type", "summary", "relevance", "sensitivity", "score", "age_days"]
            st.dataframe(pd.DataFrame([{c: x.get(c) for c in cols} for x in pk["alternatives"]]))
        tried = pk.get("tried_hooks") or []
        if tried:
            st.markdown("**Hooks and facts used so far**")
            for t in tried:
                st.markdown(f"- {t}")

    with t_evidence:
        if pk["evidence"]:
            df = pd.DataFrame(pk["evidence"])
            cols = [c for c in ["id", "category", "kind", "date", "title", "url"] if c in df.columns]
            st.dataframe(df[cols])
        else:
            st.caption("No evidence was gathered.")

    with t_log:
        st.code(result["log"], language=None)

    with t_raw:
        st.json(pk, expanded=False)


# ------------------------------------------------------------------ input tabs
tab1, tab2 = st.tabs(["Single prospect", "Batch (CSV)"])

with tab1:
    c1, c2 = st.columns(2)
    name = c1.text_input("Name", placeholder="Albinder Dhindsa")
    company = c2.text_input("Company", placeholder="Blinkit")
    title = c1.text_input("Title", placeholder="Founder and CEO")
    domain = c2.text_input("Company domain", placeholder="blinkit.com",
                           help="Strongly improves accuracy.")
    with st.expander("Optional: notes and new company"):
        new_domain = st.text_input("New company domain (only if you know they moved)",
                                   placeholder="newco.com")
        notes = st.text_area("Notes (pasted LinkedIn posts, CRM notes)", height=80)

    if st.button("Run research", type="primary", key="run_single"):
        if not (name and company):
            st.error("Name and company are required.")
        elif not key:
            st.error("Enter your Gemini API key in the sidebar.")
        elif not sender.strip():
            st.error("Enter your name under 'Your details' (it goes in the sign-off).")
        else:
            tracker, activity = progress_ui()
            st.session_state["single"] = run_one(
                {"name": name, "company": company, "title": title, "domain": domain,
                 "new_domain": new_domain, "notes": notes},
                tracker, activity, seller, find_succ)

    if "single" in st.session_state:
        st.divider()
        render(st.session_state["single"], "s", seller)

with tab2:
    st.caption("Columns: name, company, title, domain, new_domain (optional), notes, email")
    sample = "name,company,title,domain,new_domain,notes,email\nAlbinder Dhindsa,Blinkit,Founder and CEO,blinkit.com,,,\n"
    st.download_button("Download CSV template", sample, "prospects_template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        rows = list(csv.DictReader(io.StringIO(up.getvalue().decode("utf-8-sig"))))
        st.dataframe(pd.DataFrame(rows))
        if st.button("Run batch", type="primary", key="run_batch"):
            if not key:
                st.error("Enter your Gemini API key in the sidebar.")
            elif not sender.strip():
                st.error("Enter your name under 'Your details' (it goes in the sign-off).")
            else:
                results, bar = [], st.progress(0.0)
                tracker_slot, activity = st.empty(), st.empty()
                for i, row in enumerate(rows):
                    bar.progress(i / len(rows), text=f"Prospect {i + 1} of {len(rows)}: {row.get('name')}")
                    results.append(run_one(row, Tracker(tracker_slot), activity, seller, find_succ))
                bar.progress(1.0, text="Done")
                tracker_slot.empty()
                st.session_state["batch"] = results

    if "batch" in st.session_state:
        st.divider()
        res = st.session_state["batch"]

        def _pk(r):
            return r["packet"] or {}

        st.markdown("#### Summary")
        st.dataframe(pd.DataFrame([{
            "Name": _pk(r).get("prospect", {}).get("name", "-"),
            "Company": _pk(r).get("prospect", {}).get("company", "-"),
            "Status": STATUS_LABEL.get(_pk(r).get("status"), "Failed"),
            "Person status": (_pk(r).get("explanation") or {}).get("person_status"),
            "Level": _pk(r).get("level"),
            "Subject": (_pk(r).get("draft") or {}).get("subject"),
        } for r in res]))

        labels = [f"{i + 1}. {_pk(r).get('prospect', {}).get('name', 'Prospect')}" for i, r in enumerate(res)]
        sel = st.selectbox("Review prospect", range(len(res)), format_func=lambda i: labels[i])
        render(res[sel], f"b{sel}", seller)