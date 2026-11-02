"""Generate preference pairs for the Professional Email Rewriter.

Each record: prompt (same for both), chosen, rejected, rejection_type, rationale.
Rejected responses are deliberate failure modes:
  curt      - still blunt / lacks courtesy
  passive   - passive-aggressive
  verbose   - polite but bloated, buries the ask
  casual    - sloppy, unprofessional tone
Usage: python gen_data.py   -> data/train.jsonl, data/heldout.jsonl, data/demo_prompts.json
"""
import json, random, os

random.seed(42)

SYSTEM = ("You are a workplace writing assistant. Rewrite the user's message so it is "
          "professional, polite, clear and concise. Keep the original request and facts.")

NAMES = ["Priya", "Rahul", "Ananya", "Vikram", "Meera", "Arjun", "Sneha", "Karan", "Divya", "Rohan",
         "Neha", "Amit", "Pooja", "Sanjay", "Isha"]
DOCS = ["the Q3 report", "the budget sheet", "the client deck", "the test plan", "the invoice",
        "the design mockups", "the project timeline", "the onboarding doc", "the sales forecast", "the API spec"]
DAYS = ["today", "tomorrow", "Wednesday", "Friday", "end of day", "by noon", "Monday morning", "this week"]

# Each scenario: blunt templates, good templates, curt, passive-aggressive.
SCENARIOS = {
 "late_deliverable": dict(
  blunt=["{doc} is late again. get it to me {day}.", "where is {doc}?? it was due already. send it {day}."],
  good=["Hi {name}, I haven't received {doc} yet. Could you please send it {day}? Let me know if anything is blocking you.",
        "Hi {name}, could you share an update on {doc}? It was due earlier, and I'd appreciate it {day}. Happy to help if you're stuck."],
  curt=["{name}, {doc} is overdue. Send it {day}.", "Send {doc} {day}. It's late."],
  passive=["Hi {name}, just checking if {doc} is still on your radar, since it was due a while ago. Maybe {day}?",
           "Hi {name}, I know everyone's busy, but {doc} was due already. Hopefully I'll get it {day}."]),
 "reject_idea": dict(
  blunt=["your idea for {doc} won't work. do it my way.", "no. that approach to {doc} is bad. redo it."],
  good=["Hi {name}, thanks for your proposal on {doc}. I have some concerns about feasibility. Could we discuss an alternative approach?",
        "Hi {name}, I appreciate the effort on {doc}. I don't think this approach will work as is. Could we review it together and adjust?"],
  curt=["{name}, this won't work for {doc}. Redo it.", "That approach to {doc} is wrong. Change it."],
  passive=["Hi {name}, interesting attempt on {doc}. I'm sure it made sense to you, but we'll probably go another way.",
           "Hi {name}, well, I suppose that's one way to do {doc}. Let's do it properly this time."]),
 "meeting_overrun": dict(
  blunt=["stop wasting everyone's time in meetings. keep it short.", "your meetings run too long. fix it."],
  good=["Hi {name}, could we keep our meetings tighter going forward? A clear agenda and a time limit would help everyone plan their day.",
        "Hi {name}, I've noticed our meetings often run over. Would you be open to setting an agenda and sticking to the schedule?"],
  curt=["{name}, meetings are too long. Keep them short.", "Shorten your meetings."],
  passive=["Hi {name}, I'm sure we all enjoy long meetings, but some of us do have other work to finish.",
           "Hi {name}, maybe next time we could finish before the next meeting starts?"]),
 "ask_status": dict(
  blunt=["what's the status on {doc}? i need an update {day}.", "any update on {doc}? reply now."],
  good=["Hi {name}, could you share a quick status update on {doc} {day}? Thank you.",
        "Hi {name}, may I get a brief update on {doc}? I'd appreciate it {day}."],
  curt=["{name}, status of {doc}? Reply {day}.", "Update on {doc}. Now."],
  passive=["Hi {name}, I haven't heard anything on {doc}, so I assume it's going fine? Update {day} would be nice.",
           "Hi {name}, still waiting to hear on {doc}. Hopefully an update {day}."]),
 "decline_request": dict(
  blunt=["no i can't do {doc}. i'm busy.", "not my job. ask someone else about {doc}."],
  good=["Hi {name}, thank you for thinking of me for {doc}. I don't have the capacity right now. Could someone else take it, or could we revisit it later?",
        "Hi {name}, I'm unable to take on {doc} at the moment due to my current workload. I'm happy to suggest a colleague who might help."],
  curt=["{name}, I can't do {doc}. I'm busy.", "I won't be doing {doc}. Ask someone else."],
  passive=["Hi {name}, I'd love to help with {doc}, but unlike some people I have a full workload.",
           "Hi {name}, funny you ask me about {doc} when I'm already doing everyone's work."]),
 "wrong_data": dict(
  blunt=["the numbers in {doc} are wrong. check your work.", "{doc} has errors. fix them {day}."],
  good=["Hi {name}, I noticed a few figures in {doc} that don't match the source data. Could you review and correct them {day}?",
        "Hi {name}, thanks for sharing {doc}. I found some discrepancies in the numbers. Could you double-check them {day}?"],
  curt=["{name}, {doc} has errors. Fix them {day}.", "Numbers in {doc} are wrong. Check again."],
  passive=["Hi {name}, I guess nobody reviewed {doc} before sending, since the numbers are off. Fix {day}, please.",
           "Hi {name}, I'm sure the errors in {doc} were just an oversight. Please fix them {day}."]),
 "ignored_email": dict(
  blunt=["you never reply to my emails. answer me.", "i've emailed you 3 times about {doc}. respond."],
  good=["Hi {name}, I wanted to follow up on my earlier emails about {doc}. Could you please reply when you have a moment? It would help me move forward.",
        "Hi {name}, I've sent a few messages about {doc} and haven't heard back. Could you let me know when you're available to discuss?"],
  curt=["{name}, reply to my emails about {doc}.", "Answer my emails. I've sent three."],
  passive=["Hi {name}, I'm sure my emails about {doc} got lost in your busy inbox, all three of them.",
           "Hi {name}, no worries if you're too busy to reply. I'll just keep emailing about {doc}."]),
 "deadline_extension": dict(
  blunt=["i need more time for {doc}. can't make {day}.", "can't finish {doc} by {day}. extend it."],
  good=["Hi {name}, I'm running behind on {doc} and may not meet the deadline. Could we extend it by a couple of days? I'll share progress in the meantime.",
        "Hi {name}, I need a little more time to complete {doc} properly. Would an extension be possible? Thank you for understanding."],
  curt=["{name}, I can't finish {doc} by {day}. Extend it.", "Need more time for {doc}."],
  passive=["Hi {name}, given the unrealistic deadline, {doc} won't be ready. Maybe a longer one would help.",
           "Hi {name}, I'd love to finish {doc} on time, but that deadline wasn't exactly reasonable."]),
 "request_help": dict(
  blunt=["help me with {doc}. i'm stuck.", "i need you to fix {doc} for me {day}."],
  good=["Hi {name}, I'm stuck on {doc} and could use your expertise. Do you have 15 minutes {day} to help?",
        "Hi {name}, would you be able to help me with {doc} {day}? I'd really appreciate your input."],
  curt=["{name}, help with {doc}. {day}.", "Fix {doc} for me."],
  passive=["Hi {name}, since you're the expert, maybe you could finally help with {doc}?",
           "Hi {name}, I'd ask someone else about {doc}, but you're the only one who knows it."]),
 "credit_taken": dict(
  blunt=["you took credit for my work on {doc}. not ok.", "that was my work on {doc}, not yours. stop it."],
  good=["Hi {name}, I'd like to talk about how credit was shared for {doc}. I contributed significantly and would appreciate that being acknowledged. Can we chat?",
        "Hi {name}, I noticed my contribution to {doc} wasn't mentioned in the update. Could we make sure it's recognized going forward?"],
  curt=["{name}, I did the work on {doc}. Give credit.", "My work on {doc}. Not yours."],
  passive=["Hi {name}, great that {doc} got so much praise. Funny how I did most of it.",
           "Hi {name}, glad you enjoyed presenting my work on {doc}."]),
 "late_arrival": dict(
  blunt=["you're always late to standup. be on time.", "be on time for the meeting {day}. everyone's waiting."],
  good=["Hi {name}, could we all aim to join standup on time? It helps us start promptly and finish quickly. Thanks!",
        "Hi {name}, a gentle reminder to join the meeting {day} on time so we can start as scheduled. Thank you."],
  curt=["{name}, be on time {day}.", "Don't be late to standup."],
  passive=["Hi {name}, nice of you to finally join us. We managed without you.",
           "Hi {name}, I'll save you a seat, since you're usually the last to arrive."]),
 "update_docs": dict(
  blunt=["nobody updated {doc}. do it.", "{doc} is outdated. update it {day}."],
  good=["Hi {name}, {doc} seems out of date. Could you update it {day}? Please let me know if you need any information from me.",
        "Hi {name}, would you be able to refresh {doc} {day}? A few sections look outdated. Thank you."],
  curt=["{name}, update {doc} {day}.", "{doc} is outdated. Fix it."],
  passive=["Hi {name}, I guess {doc} must be a historical document, given how old it is. Update {day}?",
           "Hi {name}, someone really ought to update {doc}. {day}, maybe?"]),
 "salary_raise": dict(
  blunt=["i want a raise. i deserve more money.", "pay me more. i do more than my job."],
  good=["Hi {name}, I'd like to schedule time to discuss my compensation. Over the past year I've taken on additional responsibilities and would like to review my pay. When would suit you?",
        "Hi {name}, could we set up a meeting to discuss my role and compensation? I'd like to share my recent contributions."],
  curt=["{name}, I want a raise. Let's talk.", "I deserve higher pay. Meeting?"],
  passive=["Hi {name}, I'm sure it's an oversight that I'm doing three jobs for one salary.",
           "Hi {name}, wonder when the company will notice my work. Pay review soon?"]),
 "leave_request": dict(
  blunt=["i'm taking leave {day}. approve it.", "need leave next week. just approve."],
  good=["Hi {name}, I'd like to request leave {day}. I'll make sure my tasks are covered beforehand. Could you please approve?",
        "Hi {name}, may I take leave next week? I'll hand over my work before I go. Thank you."],
  curt=["{name}, leave {day}. Approve.", "Taking leave. Approve it."],
  passive=["Hi {name}, I'm taking leave {day}, assuming that's not a problem for anyone, as usual.",
           "Hi {name}, I know leave is a hassle for you, but I'm taking it next week."]),
 "bug_report": dict(
  blunt=["your code broke the build. fix it now.", "this bug in {doc} is your fault. fix it {day}."],
  good=["Hi {name}, the latest change appears to have broken the build. Could you take a look and fix it {day}? Happy to help debug.",
        "Hi {name}, I found a bug related to {doc}. Could you investigate it {day}? I can share the logs if useful."],
  curt=["{name}, you broke the build. Fix it now.", "Bug in {doc}. Fix {day}."],
  passive=["Hi {name}, wonderful, the build is broken again. Can you fix it {day}?",
           "Hi {name}, I'm sure the bug in {doc} wasn't tested on purpose. Fix {day}, please."]),
 "client_delay": dict(
  blunt=["we're late on {doc}. client is angry. tell them something.", "client wants {doc} now. why isn't it done?"],
  good=["Hi {name}, the client is asking about {doc}. Could you share a realistic delivery date so I can update them? Thank you.",
        "Hi {name}, the client is concerned about the delay on {doc}. Could we align on a plan and timeline today?"],
  curt=["{name}, client wants {doc}. Why isn't it done?", "Client is angry. Fix {doc}."],
  passive=["Hi {name}, the client is upset about {doc}, which I'm sure you saw coming.",
           "Hi {name}, glad the client got to experience the delay on {doc}."]),
 "noisy_office": dict(
  blunt=["stop talking loudly on calls. some of us work.", "you're too loud. shut it."],
  good=["Hi {name}, would you mind keeping your voice down during calls, or using a meeting room? It would help me concentrate. Thank you.",
        "Hi {name}, the noise during calls is distracting. Could you take them in a quiet room when possible? Thanks for understanding."],
  curt=["{name}, lower your voice on calls.", "Too loud. Take calls elsewhere."],
  passive=["Hi {name}, fascinating phone call. The whole floor heard it, as always.",
           "Hi {name}, I wasn't sure you knew we can all hear you."]),
 "review_request": dict(
  blunt=["review my pr already. it's been days.", "look at {doc}. i need feedback {day}."],
  good=["Hi {name}, could you review my pull request when you get a chance? It's been open a few days and is blocking my next task.",
        "Hi {name}, would you be able to give feedback on {doc} {day}? Thank you."],
  curt=["{name}, review my PR. It's been days.", "Review {doc} {day}."],
  passive=["Hi {name}, I'm sure you're busy with more important things than my PR, which has waited days.",
           "Hi {name}, no rush on {doc}, only that it's overdue."]),
 "scope_creep": dict(
  blunt=["stop adding new requirements. we can't do it all.", "new feature requests again? no. we're out of time."],
  good=["Hi {name}, we're at capacity for this release. Could we prioritize these new requests and schedule some for the next cycle?",
        "Hi {name}, I understand these features are important, but adding them now would put the deadline at risk. Can we discuss priorities?"],
  curt=["{name}, no more requirements. We're out of time.", "No new features. Deadline."],
  passive=["Hi {name}, sure, let's add everything. Time is infinite, after all.",
           "Hi {name}, great, more features, because we clearly have nothing else to do."]),
 "missing_info": dict(
  blunt=["you didn't send the details for {doc}. send them.", "your message makes no sense. explain."],
  good=["Hi {name}, I couldn't find the details for {doc}. Could you send them over when you can? Thank you.",
        "Hi {name}, I'm not sure I understood your last message. Could you clarify a few points?"],
  curt=["{name}, send the details for {doc}.", "Unclear message. Explain."],
  passive=["Hi {name}, I'm sure the details for {doc} were meant to be included, for once.",
           "Hi {name}, I'd love to understand your message, if it made sense."]),
}

# Scenarios reserved only for held-out evaluation (unseen situations in training)
HELDOUT_ONLY = {"scope_creep", "noisy_office", "credit_taken"}

FLUFF_OPEN = ["I hope this message finds you well and that you are having a wonderful week. ",
              "I hope you are doing great. I wanted to take a moment to reach out to you. "]
FLUFF_MID = ["I completely understand that everyone is extremely busy and that there are many competing priorities, and I really do not want to be a bother in any way. ",
             "Please do not worry at all and feel free to take whatever time you need, though, if at all possible, I was wondering whether it might perhaps be feasible to look into this. "]
FLUFF_CLOSE = [" Once again, I truly apologize for any inconvenience this may cause, and I sincerely thank you in advance for your kind time, patience and consideration.",
               " I am so sorry to trouble you, and I deeply appreciate everything you do. Please let me know if you have any questions, concerns or thoughts whatsoever."]

def sloppy(text):
    t = text.lower().replace(".", "").replace("?", "").replace("'", "")
    t = t.replace("you", "u").replace("please", "pls").replace("the ", "teh ", 1)
    return random.choice(["hey ", "yo ", "lol "]) + t + random.choice([" asap", " thx", " !!", " cmon"])

RATIONALE = {
 "curt": "Chosen opens with a greeting, frames the request politely, and offers help; rejected is clear but curt, issuing a command with no courtesy.",
 "passive": "Chosen states the request directly and respectfully; rejected hides frustration in sarcasm/passive-aggression, which damages trust.",
 "verbose": "Chosen gets to the request in 1-2 sentences; rejected is polite but bloated, burying the ask under filler and over-apologising.",
 "casual": "Chosen uses professional tone, full sentences and punctuation; rejected is slangy, lowercase and sloppy, unsuitable for workplace email.",
}

def fill(t, **kw):
    return t.format(**kw)

def make_pair(sid, sc, rtype):
    kw = dict(name=random.choice(NAMES), doc=random.choice(DOCS), day=random.choice(DAYS))
    blunt = fill(random.choice(sc["blunt"]), **kw)
    good = fill(random.choice(sc["good"]), **kw)
    if rtype == "curt": rej = fill(random.choice(sc["curt"]), **kw)
    elif rtype == "passive": rej = fill(random.choice(sc["passive"]), **kw)
    elif rtype == "verbose":
        rej = random.choice(FLUFF_OPEN) + random.choice(FLUFF_MID) + good + random.choice(FLUFF_CLOSE)
    else: rej = sloppy(blunt)
    prompt = f"Rewrite this workplace message professionally, politely and concisely:\n\n\"{blunt}\""
    return dict(scenario=sid, prompt=prompt, chosen=good, rejected=rej, rejection_type=rtype,
                rationale=RATIONALE[rtype], system=SYSTEM)

def build(scen_ids, n):
    out, seen, tries = [], set(), 0
    types = ["curt", "passive", "verbose", "casual"]
    while len(out) < n and tries < n * 50:
        tries += 1
        sid = random.choice(scen_ids)
        p = make_pair(sid, SCENARIOS[sid], types[len(out) % 4])
        key = (p["prompt"], p["rejected"])
        if key in seen: continue
        seen.add(key); out.append(p)
    return out

def main():
    os.makedirs("data", exist_ok=True)
    train_ids = [s for s in SCENARIOS if s not in HELDOUT_ONLY]
    train = build(train_ids, 640)
    # held-out: 40 from seen scenarios (new prompts) + 20 from fully unseen scenarios
    seen_keys = {p["prompt"] for p in train}
    held = [p for p in build(train_ids, 120) if p["prompt"] not in seen_keys][:40]
    held += build(sorted(HELDOUT_ONLY), 20)
    for name, rows in [("train", train), ("heldout", held)]:
        with open(f"data/{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    demo = [
     "hey the vendor still hasn't sent the contract. chase them or i will.",
     "your slides were a mess. redo them before the board meeting tmrw.",
     "i can't cover your shift again. stop asking me.",
     "why was i not invited to the planning call? that's unacceptable.",
     "ur estimate is way too high. cut it by half or we're done.",
    ]
    json.dump(demo, open("data/demo_prompts.json", "w"), indent=1)
    print(f"train={len(train)} heldout={len(held)} demo={len(demo)}")

if __name__ == "__main__":
    main()
