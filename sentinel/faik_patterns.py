"""FAIK-derived scam detection patterns for JobSentinel.

Patterns extracted from:
  "FAIK: A Practical Guide to Living in a World of Deepfakes, Disinformation,
  and AI-Generated Deceptions" by Perry Carpenter (Wiley, 2024).

Key source chapters:
  Ch. 3 — Deceptionology 101: cognitive biases, framing, social engineering
  Ch. 6 — Deepfakes and the Spectrum of Digital Deception
  Ch. 7 — The Now and Future of AI-Driven Deception (AI-powered scams)
  Ch. 9 — Cognitive and Technical Defense Strategies

Core FAIK framework applied here:
  - Deception operates on FACTS + FRAMES: scammers curate what information
    you see (facts) and manipulate the context you use to interpret it (frames).
  - We live in the "exploitation zone" where AI levels the playing field for
    scammers — any low-skill fraudster can now generate perfect, personalized,
    grammatically-flawless job postings, emails, and voice calls.
  - Scammers target System 1 (fast, automatic) thinking: urgency, authority,
    and emotional triggers bypass rational scrutiny.
  - AI-powered schemes "adapt on-the-fly" based on victim responses —
    watch for escalating trust-building followed by a sudden ask.

These patterns extend knowledge.py's _DEFAULT_PATTERNS with signals
specifically motivated by the deepfake/AI-scam threat model.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# FAIK-derived seed patterns
# Each dict matches the _DEFAULT_PATTERNS schema in knowledge.py:
#   pattern_id, name, description, category, regex, keywords
# ---------------------------------------------------------------------------

FAIK_PATTERNS: list[dict] = [
    # -----------------------------------------------------------------------
    # AI-GENERATED JOB POSTINGS
    # Ch. 7: "AI has leveled the playing field to the point where any
    # scammer — regardless of skill level or location — can create a
    # world-class phishing email in any language, mimicking any brand."
    # The tell: AI-generated postings are suspiciously polished, use
    # generic filler superlatives, and mix vague role descriptions with
    # exact salary figures to seem credible.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_ai_generated_posting",
        "name": "AI-Generated Job Posting Markers",
        "description": (
            "Linguistic patterns characteristic of LLM-generated job postings: "
            "filler superlatives, suspiciously smooth prose, and generic "
            "boilerplate that appears professional but contains no specifics. "
            "FAIK Ch.7: AI allows any scammer to produce world-class-looking "
            "postings regardless of skill."
        ),
        "category": "warning",
        "regex": (
            r"(?i)(dynamic\s+(and\s+)?motivated\s+(individual|professional|team\s+player)"
            r"|passionate\s+about\s+(making\s+a\s+difference|innovation|growth|impact)"
            r"|fast.?paced\s+(environment|startup|company).{0,60}(opportunity|role|position)"
            r"|join\s+our\s+(growing|dynamic|innovative|passionate)\s+team"
            r"|we\s+are\s+looking\s+for\s+a\s+(talented|driven|motivated|passionate)\s+\w+"
            r"|competitive\s+salary\s+and\s+benefits\s+package)"
        ),
        "keywords": [
            "dynamic and motivated individual",
            "passionate about making a difference",
            "join our growing team",
            "fast-paced environment",
            "competitive salary and benefits package",
            "talented and driven professional",
        ],
    },

    # -----------------------------------------------------------------------
    # DEEPFAKE / AI VIDEO INTERVIEW SCAMS
    # Ch. 6 + Ch. 7: Scammers use AI voice cloning and deepfake video to
    # conduct fake "interviews." The FBI (and FAIK) document cases of
    # candidates interviewing via video with an AI-generated hiring manager.
    # Red flags: interview conducted entirely via chat/text, interviewer
    # declines video, audio/video quality issues described as tech problems,
    # or "pre-recorded" interview formats used to avoid live interaction.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_deepfake_interview_vector",
        "name": "Deepfake / Synthetic Interview Format",
        "description": (
            "Interview conducted via one-way video, chat-only, or pre-recorded "
            "format — eliminating live human interaction and enabling AI/deepfake "
            "impersonation. FAIK Ch.6+7: voice cloning can produce a convincing "
            "hiring manager from minutes of public audio; deepfake video "
            "interviews require only a brief video sample."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(one.?way\s+video\s+interview"
            r"|pre.?recorded\s+(video\s+)?interview"
            r"|interview\s+via\s+(text|chat|whatsapp|telegram|signal|google\s+chat)"
            r"|no\s+(video|camera)\s+(required|needed|necessary)"
            r"|interviewer\s+(camera|video)\s+(off|disabled|not\s+(working|available))"
            r"|asynchronous\s+(video\s+)?interview)"
        ),
        "keywords": [
            "one-way video interview",
            "pre-recorded video interview",
            "interview via text chat",
            "interview via WhatsApp",
            "no camera required for interview",
            "asynchronous interview",
            "interviewer camera off",
        ],
    },

    # -----------------------------------------------------------------------
    # VOICE CLONING / VISHING RECRUITMENT
    # Ch. 7: "With AI voice cloning, a scammer can sound just like your
    # bank teller, your boss, even your mom." In job scams, vishing calls
    # claim to be from HR departments or recruiters at real companies.
    # Carpenter's ScamBot demo used Kevin Mitnick's cloned voice to
    # extract information in real-time phone conversations.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_voice_cloning_recruitment",
        "name": "Suspicious Unsolicited Recruiter Call",
        "description": (
            "Unsolicited phone/voice recruitment with immediate offer or "
            "sensitive data request. FAIK Ch.7: AI voice cloning lets "
            "scammers impersonate real company HR staff using publicly "
            "available audio — podcast appearances, YouTube videos, "
            "LinkedIn audio profiles — to build instant trust."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(called\s+you\s+directly|reached\s+out\s+by\s+phone"
            r"|our\s+(HR|recruiter|hiring\s+manager)\s+will\s+call\s+you"
            r"|phone\s+(screen|interview)\s+(today|immediately|right\s+away|asap)"
            r"|we\s+found\s+your\s+(resume|profile)\s+and\s+(want|would\s+like)\s+to\s+(offer|hire)"
            r"|skip\s+the\s+(interview\s+process|application|screening))"
        ),
        "keywords": [
            "our recruiter will call you today",
            "we found your resume and want to hire you",
            "phone screen immediately",
            "skip the interview process",
            "called you directly with offer",
        ],
    },

    # -----------------------------------------------------------------------
    # SOCIAL ENGINEERING ESCALATION / TRUST-BUILDING THEN ASK
    # Ch. 3 + Ch. 7: Carpenter's "deceptionology" framework describes how
    # scams manipulate FACTS + FRAMES to build trust before the ask.
    # The pattern: establish legitimacy (fake company details, real-looking
    # website, employee LinkedIn profiles) → build rapport over multiple
    # touchpoints → sudden pivot to request for money/data/action.
    # Multi-touch campaigns that feel "too good to be true" before pivoting.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_trust_escalation_pivot",
        "name": "Trust-Building Escalation Before Sensitive Ask",
        "description": (
            "Progressive trust-building sequence (multiple warm contact points, "
            "flattery, exclusive framing) followed by a financial or data ask. "
            "FAIK Ch.3: 'deceptionologists' exploit System 1 thinking by "
            "establishing trust frames before the actual exploit — the ask "
            "feels natural by the time it arrives."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(you\s+(have\s+been\s+)?selected\s+(from\s+(thousands|hundreds)|exclusively)"
            r"|we\s+(have\s+)?reviewed\s+your\s+profile\s+and\s+(you('re|\s+are)\s+(a\s+)?(perfect|ideal|exceptional|top)\s+(fit|candidate|match))"
            r"|this\s+(is\s+a\s+)?(confidential|exclusive|private)\s+(opportunity|offer|position)"
            r"|before\s+we\s+proceed\s+(further|to\s+the\s+next\s+step).{0,80}(provide|send|submit|verify)"
            r"|next\s+step\s+is\s+to\s+(send|provide|submit|pay|deposit|purchase))"
        ),
        "keywords": [
            "selected from thousands of candidates",
            "you are the perfect fit",
            "confidential opportunity",
            "exclusive position for you",
            "before we proceed please provide",
            "next step is to send payment",
            "you have been exclusively selected",
        ],
    },

    # -----------------------------------------------------------------------
    # IMPERSONATION OF KNOWN LEGITIMATE COMPANIES
    # Ch. 7: "AI deepfakes can sound just like your bank teller, your boss"
    # Ch. 6: Scammers weaponize real brand recognition (halo effect) to
    # lower victim defenses — Ch. 3 "halo effect" cognitive bias.
    # A job posting that invokes a famous company but uses non-corporate
    # contact channels is a primary signal.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_brand_impersonation_job",
        "name": "Brand Impersonation in Job Posting",
        "description": (
            "Job claims affiliation with a well-known company but uses personal "
            "email domains or unofficial communication channels. FAIK Ch.3: the "
            "'halo effect' bias — a recognizable brand name lowers skepticism. "
            "Scammers exploit this by pairing a famous company name with "
            "low-credibility contact channels."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(hiring\s+(for|on\s+behalf\s+of)\s+(google|amazon|microsoft|apple|meta|facebook|netflix|tesla|openai|anthropic|nvidia|deloitte|mckinsey|goldman\s+sachs))"
            r"|(google|amazon|microsoft|apple|meta|facebook).{0,60}(@gmail\.com|@yahoo\.com|@hotmail\.com|@outlook\.com)"
            r"|(on\s+behalf\s+of|representing|contracted\s+(by|for)).{0,60}(google|amazon|microsoft|apple|meta|facebook|netflix)"
        ),
        "keywords": [
            "hiring for Google via gmail",
            "on behalf of Amazon HR",
            "contracted by Microsoft",
            "representing Apple recruiting",
            "hiring for Meta via personal email",
        ],
    },

    # -----------------------------------------------------------------------
    # AI CHATBOT RECRUITMENT (SMISHING / CHAT-BASED LURE)
    # Ch. 7: "Smishing — AI lets scammers set up chatbots that can carry
    # on a full-blown text conversation. It feels like a human on the other
    # end. Spoiler alert: it's not."
    # Ch. 7 (ScamBot demo): Carpenter built bots that engaged in real-time
    # voice + text conversations indistinguishable from humans.
    # Signals: initial contact via SMS/WhatsApp/Telegram, not email/LinkedIn.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_chatbot_recruitment_channel",
        "name": "Non-Standard Channel Initial Recruitment Contact",
        "description": (
            "Job opportunity delivered via SMS, WhatsApp, Telegram, or other "
            "non-professional messaging platform. FAIK Ch.7: AI chatbots can "
            "sustain a full conversation indistinguishable from human recruiters "
            "over text — automating hundreds of simultaneous 'recruitment' "
            "conversations at near-zero marginal cost."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(contact\s+(us|me|our\s+team)\s+(via|on|through|at)\s+(whatsapp|telegram|signal|wechat|viber|line\s+app))"
            r"|(reach\s+(us|out)\s+(via|on|through|at)\s+(whatsapp|telegram|signal))"
            r"|(apply\s+(via|through|on)\s+(whatsapp|telegram|text\s+message|sms))"
            r"|(send\s+(your\s+)?(cv|resume|application)\s+(to\s+)?(\+\d{10,}|via\s+whatsapp|via\s+telegram))"
        ),
        "keywords": [
            "contact us via WhatsApp",
            "apply through Telegram",
            "send your CV via WhatsApp",
            "reach out on Telegram for details",
            "contact recruiter on Signal",
            "apply via text message",
        ],
    },

    # -----------------------------------------------------------------------
    # SYNTHETIC IDENTITY / FAKE COMPANY RED FLAGS
    # Ch. 7: "Synthetic identity fraud — AI can create fake people with
    # backstories so detailed they could be your next-door neighbor."
    # Fake companies have: no verifiable address, very recently created
    # web presence, suspiciously generic descriptions, and often claim
    # to be "international" or "global" with no traceable offices.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_synthetic_company_identity",
        "name": "Synthetic / Unverifiable Company Identity",
        "description": (
            "Company details that resist independent verification: no physical "
            "address, generic 'global staffing' descriptions, very recent "
            "founding with suspiciously large claims, or mismatch between "
            "stated size and actual digital footprint. FAIK Ch.7: AI enables "
            "creation of synthetic company identities complete with fake "
            "employee LinkedIn profiles, AI-generated headshots, and "
            "fabricated press releases."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(global\s+(staffing|consulting|solutions|recruitment)\s+(firm|company|agency)"
            r".{0,120}(no\s+location|remote\s+only|location\s+not\s+specified))"
            r"|(we\s+are\s+a\s+(leading|top|premier)\s+(global|international|worldwide)\s+(company|firm|agency)"
            r".{0,200}(founded\s+in\s+20(2[0-9])|established\s+20(2[0-9])))"
            r"|(international\s+(recruitment|staffing|placement)\s+agency.{0,80}(no\s+(website|address|office)))"
        ),
        "keywords": [
            "global staffing firm no location",
            "leading international company founded 2023",
            "premier global agency no address",
            "worldwide recruitment no office listed",
            "top global solutions company established 2024",
        ],
    },

    # -----------------------------------------------------------------------
    # EMOTIONAL MANIPULATION / FEAR-HOPE EXPLOITATION
    # Ch. 3: "Deceptionologists know that we humans are not the logical,
    # rational beings we like to think we are... emotional triggers can be
    # easily exploited by those who know how to push the right buttons."
    # Ch. 7: Scammers craft appeals to "deepest desires and darkest fears."
    # Job scam emotional levers: fear of unemployment, hope of escape,
    # appeal to desperation ("we understand your situation").
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_emotional_exploitation_job",
        "name": "Emotional Vulnerability Exploitation in Job Ad",
        "description": (
            "Language designed to exploit System 1 emotional triggers: "
            "appeals to desperation, financial fear, or the hope of dramatic "
            "life improvement. FAIK Ch.3: 'deceptionologists' target "
            "emotional vulnerabilities — scammers leverage unemployment anxiety "
            "and financial pressure to bypass rational scrutiny of the offer."
        ),
        "category": "warning",
        "regex": (
            r"(?i)(tired\s+of\s+(your\s+(current\s+)?(job|situation|boss|life)|struggling|being\s+(broke|underpaid))"
            r"|change\s+your\s+(life|financial\s+situation|future)\s+(today|now|forever)"
            r"|finally\s+(be|become)\s+(financially\s+(free|independent)|your\s+own\s+boss)"
            r"|we\s+understand\s+(how\s+(hard|difficult|tough)|your\s+(struggle|situation|frustration))"
            r"|stop\s+(living\s+paycheck\s+to\s+paycheck|struggling\s+financially))"
        ),
        "keywords": [
            "tired of your current job",
            "change your life today",
            "finally be financially free",
            "we understand your struggle",
            "stop living paycheck to paycheck",
            "become your own boss",
            "tired of being underpaid",
        ],
    },

    # -----------------------------------------------------------------------
    # SPEAR PHISHING JOB CONTACT (PERSONALIZED LURE)
    # Ch. 7: "Spear phishing — these hyper-targeted attacks use every bit
    # of dirt the AI can dig up on you from social media and beyond. It's
    # like they've got a secret dossier on your whole life."
    # In job scams: AI scrapes LinkedIn, GitHub, or portfolio sites to
    # craft hyper-personalized outreach that references real details
    # about the victim's career history, skills, or recent activity.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_spear_phishing_job_outreach",
        "name": "Hyper-Personalized Job Outreach (Spear Phishing)",
        "description": (
            "Unsolicited job outreach that references specific details about "
            "the recipient's career history, skills, or recent public activity "
            "in ways suggesting automated OSINT scraping. FAIK Ch.7: AI-powered "
            "spear phishing builds hyper-targeted dossiers from social media "
            "to make fraudulent outreach feel genuine and personally relevant."
        ),
        "category": "warning",
        "regex": (
            r"(?i)(we\s+(noticed|saw|found|came\s+across)\s+your\s+(recent\s+)?(post|article|project|work|profile)\s+(on|about|regarding))"
            r"|(your\s+(background|experience|skills?)\s+in\s+\w+.{0,60}(caught|impressed|perfect\s+match|exactly\s+what\s+we))"
            r"|(based\s+on\s+your\s+(linkedin|github|portfolio|recent\s+work).{0,80}(reach\s+out|contact(ed)?\s+you|this\s+opportunity))"
        ),
        "keywords": [
            "we noticed your recent post",
            "your background in X caught our attention",
            "based on your LinkedIn profile we are reaching out",
            "your experience is exactly what we need",
            "we came across your portfolio",
            "impressed by your recent project",
        ],
    },

    # -----------------------------------------------------------------------
    # LIAR'S DIVIDEND / DEEPFAKE DISCLAIMER ABUSE
    # Ch. 7: "Undermining trust and sowing confusion — bad actors could
    # claim real videos are fake to sow doubt. This is the liar's dividend
    # at work — allowing anyone to avoid accountability."
    # In job scams: fraudulent recruiters pre-emptively warn about
    # "fake job scams" to appear trustworthy, or invoke deepfake awareness
    # to explain away any inconsistencies in their own video calls.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_liars_dividend_trust_signal",
        "name": "Preemptive Scam-Warning as Trust Signal",
        "description": (
            "Fraudulent recruiter proactively warns about 'fake job scams' or "
            "invokes deepfake awareness to appear legitimate. FAIK Ch.7: the "
            "'liar's dividend' — bad actors weaponize public awareness of "
            "deepfakes/scams to preemptively inoculate victims against skepticism, "
            "using meta-awareness of fraud as a trust-building frame."
        ),
        "category": "warning",
        "regex": (
            r"(?i)(we\s+are\s+(not\s+a\s+scam|100%\s+legitimate|a\s+real\s+company|verified)"
            r"|beware\s+of\s+(fake|fraudulent|scam)\s+(job\s+)?(postings?|offers?|recruiters?).{0,120}(we\s+are\s+(different|legitimate|real))"
            r"|unlike\s+(other|fake|scam)\s+(recruiters?|companies?|postings?).{0,80}(we|our\s+(company|offer|process))"
            r"|this\s+is\s+(a\s+)?(100%\s+)?(legitimate|real|verified|genuine)\s+(opportunity|offer|job|position))"
        ),
        "keywords": [
            "we are not a scam",
            "beware of fake job postings unlike us",
            "100% legitimate opportunity",
            "verified real company",
            "unlike other fake recruiters we are genuine",
            "this is a real verified job offer",
        ],
    },

    # -----------------------------------------------------------------------
    # FAST-MOVING OFFER / OODA LOOP HIJACKING
    # Ch. 3 + Ch. 7: Carpenter references the OODA Loop (Observe-Orient-
    # Decide-Act) as a target for manipulation. Scammers compress the
    # Decide→Act steps by manufacturing false urgency, preventing victims
    # from completing the Orient step (research, skepticism).
    # Ch. 3: "Cunning deceivers target our intuitive System 1 by
    # heightening emotions, urgency, and curiosity."
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_ooda_loop_hijack",
        "name": "Decision-Compression / OODA Loop Hijack",
        "description": (
            "Artificial time pressure that forces a decision before the "
            "candidate can research the company or role. FAIK Ch.3: "
            "scammers hijack the OODA Loop by collapsing the Orient step — "
            "research, verification, and skepticism — via fake deadlines, "
            "competing-offer pressure, and same-day decision demands."
        ),
        "category": "warning",
        "regex": (
            r"(?i)(offer\s+(is\s+)?(only\s+)?valid\s+(for|until|through)\s+(24|48|72)\s+hours?"
            r"|must\s+(accept|decide|confirm|respond)\s+(within|by)\s+(24|48|72)\s+hours?"
            r"|we\s+have\s+(another|other)\s+candidate(s)?\s+(interested|waiting|in\s+consideration)"
            r"|position\s+(will\s+be\s+)?filled\s+(immediately|today|this\s+week)\s+if\s+(you\s+)?(don't|do\s+not)\s+(respond|accept|confirm)"
            r"|this\s+offer\s+expires\s+(today|tonight|at\s+midnight|in\s+\d+\s+hours?))"
        ),
        "keywords": [
            "offer valid for 24 hours only",
            "must accept within 48 hours",
            "we have another candidate waiting",
            "position will be filled today",
            "this offer expires tonight",
            "decide by end of day",
        ],
    },

    # -----------------------------------------------------------------------
    # AI-ASSISTED ROMANCE-TO-JOB SCAM CROSSOVER
    # Ch. 7: "Dating app scams — fully automated, allowing one scammer
    # to run hundreds of automated profiles capable of sending messages,
    # voice memos, candid pics, and even real-time video chats."
    # Romance scams increasingly pivot to fake "job opportunities" once
    # initial trust is established — the romantic connection becomes the
    # trust frame for the financial ask.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_romance_to_job_pivot",
        "name": "Romance / Relationship to Job Opportunity Pivot",
        "description": (
            "Romantic or friendly connection that transitions into a job offer, "
            "investment opportunity, or 'side business' pitch. FAIK Ch.7: "
            "AI chatbots sustain extended romantic relationships before pivoting "
            "to financial exploitation — the cultivated trust becomes a "
            "deception frame for the economic ask."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(my\s+(friend|contact|colleague|partner).{0,60}(job|opportunity|opening|position|hiring)"
            r"|met\s+(online|on\s+(a\s+)?(dating\s+app|instagram|facebook|tinder|hinge)).{0,120}(job|work|opportunity|business)"
            r"|I\s+(thought|wanted)\s+to\s+(share|tell\s+you\s+about)\s+this\s+(amazing|great|incredible)\s+(opportunity|job|investment)"
            r"|my\s+(boss|manager|company)\s+(is\s+)?(looking\s+for|hiring|needs)\s+someone\s+like\s+you)"
        ),
        "keywords": [
            "my friend has a job for you",
            "met online and mentioned a job opportunity",
            "wanted to share this amazing opportunity",
            "my boss is looking for someone like you",
            "great side business my contact told me about",
        ],
    },

    # -----------------------------------------------------------------------
    # CHEAPFAKE / OUT-OF-CONTEXT COMPANY EVIDENCE
    # Ch. 6: "Cheapfakes — low-tech manipulation methods like editing,
    # splicing, or altering context of genuine footage — can be devastating."
    # Ch. 6: "Sometimes the cheapest of fakes is simply a piece of media
    # and a lie." Scammers reuse real company photos, real employee names,
    # and real press releases stripped of context to build fake credibility.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_cheapfake_company_credibility",
        "name": "Out-of-Context / Reused Credibility Claims",
        "description": (
            "Company or recruiter uses real-seeming but unverifiable credibility "
            "markers: awards from unspecified organizations, partnerships with "
            "unnamed Fortune 500 companies, screenshots of testimonials, or "
            "claims of media features. FAIK Ch.6: 'cheapfakes' — recontextualized "
            "real content — are more common and harder to detect than sophisticated "
            "deepfakes, and equally effective at building false legitimacy."
        ),
        "category": "warning",
        "regex": (
            r"(?i)(featured\s+in\s+(forbes|inc\.|entrepreneur|business\s+insider|techcrunch).{0,80}(recently|last\s+year|this\s+year))"
            r"|(award.?winning\s+(company|team|firm|agency).{0,60}(best|top|leading|premier))"
            r"|(trusted\s+by\s+(fortune\s+500|hundreds|thousands|millions)\s+of\s+(companies|clients|businesses))"
            r"|(partnered\s+with\s+(leading|major|top)\s+(companies|brands|corporations)\s+(including|such\s+as))"
        ),
        "keywords": [
            "featured in Forbes recently",
            "award-winning company trusted by Fortune 500",
            "trusted by thousands of companies",
            "partnered with leading major brands",
            "inc 5000 company",
            "featured in TechCrunch",
        ],
    },

    # -----------------------------------------------------------------------
    # DATA HARVESTING UNDER RECRUITMENT COVER
    # Ch. 7: "Bad actors mining LLMs for personal data" + spear phishing
    # via AI. Fake job applications are a primary vector for harvesting
    # PII: SSN, passport, bank details, and identity documents requested
    # as part of a plausible "background check" or "onboarding" workflow.
    # FAIK distinguishes this from blunt SSN requests (already in
    # knowledge.py) — here the harvest is spread across a multi-step
    # onboarding flow that feels legitimate.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_onboarding_data_harvest",
        "name": "Fake Onboarding / Multi-Step PII Harvest",
        "description": (
            "Legitimate-seeming onboarding flow that requests PII "
            "(SSN, ID scans, bank account) before any formal offer letter, "
            "contract, or verifiable company documentation. FAIK Ch.7: "
            "scammers now run multi-step data harvesting campaigns disguised "
            "as professional HR onboarding, using AI-generated documents "
            "and portals to build a convincing fictional employer."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(complete\s+(your\s+)?(onboarding|new\s+hire|pre.?employment)\s+(before|prior\s+to)\s+(receiving|signing|getting)\s+(your\s+)?(offer|contract|letter))"
            r"|(upload\s+(your\s+)?(passport|driver.?s\s+license|id|identification)\s+(to\s+(our\s+)?portal|before\s+the\s+interview))"
            r"|(provide\s+(your\s+)?(banking|bank\s+account|direct\s+deposit)\s+information\s+(to\s+)?(reserve|secure|hold)\s+(your\s+)?(spot|position|seat))"
            r"|(background\s+check.{0,60}(pay|fee|cost|charge|purchase))"
        ),
        "keywords": [
            "complete onboarding before receiving offer letter",
            "upload passport to our portal",
            "provide banking information to reserve your spot",
            "background check fee required",
            "upload ID before interview",
            "direct deposit info needed before offer",
        ],
    },

    # -----------------------------------------------------------------------
    # SYSTEM 1 AUTHORITY TRIGGER (IMPERSONATING AUTHORITY FIGURES)
    # Ch. 3: "Social engineers exploit mental shortcuts, like our deep-
    # seated urge to help authority figures in times of crisis. Even savvy
    # cybersecurity pros can fall victim when malicious hackers wield
    # these cognitive weapons."
    # In job context: impersonating C-suite, government agencies, or
    # well-known executives to lend authority to a job offer.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_authority_impersonation",
        "name": "Fake Authority Figure in Recruitment",
        "description": (
            "Recruiter claims to be or represent a C-suite executive, "
            "government hiring agency, or named public figure without "
            "verifiable institutional contact channels. FAIK Ch.3: "
            "cognitive bias toward authority figures ('authority heuristic') "
            "suppresses skepticism — scammers exploit this by invoking "
            "recognizable titles and names to manufacture instant credibility."
        ),
        "category": "warning",
        "regex": (
            r"(?i)(on\s+behalf\s+of\s+(CEO|CTO|CFO|President|Director|VP)\s+\w+)"
            r"|(personally\s+(selected|recommended|referred)\s+by\s+(our\s+)?(CEO|CTO|founder|president|director))"
            r"|(government.{0,40}(hiring|recruitment|opportunity).{0,80}(federal|agency|department|bureau))"
            r"|(referred\s+to\s+you\s+by\s+(CEO|CTO|CFO|executive|senior\s+(vice\s+)?president))"
        ),
        "keywords": [
            "on behalf of CEO personally selected you",
            "personally recommended by our founder",
            "government hiring program federal agency",
            "referred by the CEO",
            "director personally reviewed your profile",
        ],
    },

    # -----------------------------------------------------------------------
    # AI-GENERATED SYNTHETIC MEDIA IN APPLICATION PROCESS
    # Ch. 6: Deepfakes and AI-generated images are increasingly used to
    # create fake employee headshots (LinkedIn), fake company office photos,
    # and fabricated team pages that make fraudulent companies look real.
    # The "IYKYK effect" — it's hard to spot fakes unless you're already
    # suspicious and looking for them.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_synthetic_recruiter_identity",
        "name": "Unverifiable Recruiter Identity / AI-Generated Profile",
        "description": (
            "Recruiter or hiring manager whose identity cannot be verified "
            "through LinkedIn, the company website, or professional databases — "
            "or whose profile shows signs of AI-generated content (generic "
            "headshots, no work history, recent account creation). FAIK Ch.6: "
            "AI image generation creates convincing fake employee profiles; "
            "most people cannot detect AI-generated faces without prior warning."
        ),
        "category": "warning",
        "regex": (
            r"(?i)(recruiter\s+(has\s+)?(no|zero|0)\s+(linkedin|social\s+media|online)\s+(profile|presence|account))"
            r"|(contact\s+(person|recruiter|hr).{0,80}(not\s+(found|listed|verified)|cannot\s+be\s+(found|verified|confirmed)))"
            r"|(hiring\s+(contact|manager|person)\s+(email|name)\s+(does\s+not\s+match|different\s+from)\s+(company|website|domain))"
        ),
        "keywords": [
            "recruiter has no LinkedIn profile",
            "contact person cannot be verified",
            "hiring manager not found online",
            "recruiter email doesn't match company domain",
            "no verifiable online presence",
        ],
    },

    # -----------------------------------------------------------------------
    # INVESTMENT / CRYPTO JOB CROSSOVER (PIG BUTCHERING)
    # Ch. 7: "Cryptocurrency scams — with AI chatbots shilling the latest
    # and greatest digital Ponzi schemes."
    # Ch. 7 (Romance to Job pivot): The "pig butchering" scam pattern
    # combines romance/friendship trust-building with a fake crypto trading
    # platform "job" — victims are asked to "invest" as part of their
    # job training or onboarding.
    # -----------------------------------------------------------------------
    {
        "pattern_id": "faik_pig_butchering_job",
        "name": "Pig Butchering / Crypto Investment Job Scam",
        "description": (
            "Job that involves crypto trading, investment platform operation, "
            "or financial asset management as a core duty — especially when "
            "the role requires the applicant to deposit funds or 'demonstrate "
            "trading ability' using their own money. FAIK Ch.7: pig-butchering "
            "combines romantic/social trust-building with a fake crypto "
            "trading platform posing as employer training."
        ),
        "category": "red_flag",
        "regex": (
            r"(?i)(crypto(currency)?\s+(trader|trading|investment)\s+(job|role|position|opportunity))"
            r"|(forex\s+(trader|trading)\s+(from\s+home|remote|work.?from.?home))"
            r"|(demonstrate\s+(your\s+)?(trading|investment)\s+(skills?|ability|knowledge).{0,80}(deposit|fund|invest|own\s+money))"
            r"|(our\s+(platform|system|app).{0,60}(start\s+with|initial|deposit|fund|invest)\s+\$)"
            r"|(earn\s+(commission|profit|returns?)\s+(on\s+)?(your\s+)?(trades?|investments?|portfolio))"
        ),
        "keywords": [
            "crypto trading job from home",
            "forex trading remote position",
            "demonstrate trading skills with your own deposit",
            "start with initial investment on our platform",
            "earn commission on your trades",
            "crypto investment opportunity as employment",
        ],
    },
]


def get_faik_pattern_ids() -> list[str]:
    """Return the list of pattern_id strings from FAIK_PATTERNS."""
    return [p["pattern_id"] for p in FAIK_PATTERNS]
