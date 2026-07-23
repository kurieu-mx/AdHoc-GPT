"""A synthetic corpus of diplomatic documents: resolutions, debate, position papers.

Phase 3 needs domain text and there is no redistributable, pre-packaged corpus of
UN-style drafting. This module *generates* one from grammars built out of the
conventional register: preambular participles, operative verbs, procedural
formulae. Everything here is synthetic and templated -- it is a stylistic
target, not a factual record. Do not treat generated documents as real UN
material, and do not train a model on this and then present its output as
authentic practice.

    python -m adhoc_gpt.domain.corpus --out data/raw/diplomacy.txt --docs 6000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# --- document scaffolding -------------------------------------------------
ORGANS = [
    ("The General Assembly", "GENERAL ASSEMBLY", "A/RES"),
    ("The Security Council", "SECURITY COUNCIL", "S/RES"),
    ("The Economic and Social Council", "ECONOMIC AND SOCIAL COUNCIL", "E/RES"),
    ("The Human Rights Council", "HUMAN RIGHTS COUNCIL", "A/HRC/RES"),
    ("The Ad Hoc Committee", "AD HOC COMMITTEE", "AC/RES"),
]

COMMITTEES = [
    "First Committee (Disarmament and International Security)",
    "Second Committee (Economic and Financial)",
    "Third Committee (Social, Humanitarian and Cultural)",
    "Fourth Committee (Special Political and Decolonization)",
    "Sixth Committee (Legal)",
    "Ad Hoc Committee on Emerging Technologies",
]

DELEGATIONS = [
    "Argentina", "Australia", "Bangladesh", "Botswana", "Brazil", "Canada",
    "Chile", "Costa Rica", "Denmark", "Egypt", "Estonia", "Ethiopia", "Fiji",
    "France", "Germany", "Ghana", "Iceland", "India", "Indonesia", "Ireland",
    "Japan", "Jordan", "Kenya", "Malaysia", "Mexico", "Mongolia", "Morocco",
    "Nepal", "the Netherlands", "New Zealand", "Nigeria", "Norway", "Peru",
    "the Philippines", "Portugal", "Rwanda", "Senegal", "Singapore", "Slovenia",
    "South Africa", "the Republic of Korea", "Spain", "Sweden", "Switzerland",
    "Thailand", "Tunisia", "Uruguay", "Viet Nam", "Zambia",
]

BLOCS = [
    "the Group of 77 and China", "the African Group", "the Nordic countries",
    "the Alliance of Small Island States", "the European Union",
    "the Non-Aligned Movement", "the least developed countries",
    "the landlocked developing countries",
]

# --- topics: each supplies the noun phrases the clauses are built from ----
TOPICS: dict[str, dict[str, list[str]]] = {
    "climate resilience": {
        "issue": ["the accelerating loss of coastal land", "climate-induced displacement",
                  "the widening adaptation finance gap", "recurrent drought in vulnerable regions"],
        "instrument": ["the Paris Agreement", "the Sendai Framework for Disaster Risk Reduction",
                       "the United Nations Framework Convention on Climate Change"],
        "measure": ["early warning systems", "concessional adaptation finance",
                    "nature-based coastal defences", "national adaptation plans"],
        "actor": ["small island developing States", "affected coastal communities",
                  "national meteorological services"],
    },
    "nuclear disarmament": {
        "issue": ["the continued existence of nuclear arsenals", "the risk of accidental launch",
                  "the erosion of arms control architecture", "the resumption of testing"],
        "instrument": ["the Treaty on the Non-Proliferation of Nuclear Weapons",
                       "the Comprehensive Nuclear-Test-Ban Treaty",
                       "the Treaty on the Prohibition of Nuclear Weapons"],
        "measure": ["verifiable and irreversible reductions", "negative security assurances",
                    "de-alerting measures", "transparency in fissile material stocks"],
        "actor": ["nuclear-weapon States", "the Preparatory Commission",
                  "regional nuclear-weapon-free zones"],
    },
    "refugee protection": {
        "issue": ["protracted displacement", "the criminalization of rescue at sea",
                  "shrinking humanitarian access", "the strain on host communities"],
        "instrument": ["the 1951 Convention relating to the Status of Refugees",
                       "the Global Compact on Refugees", "the principle of non-refoulement"],
        "measure": ["responsibility-sharing arrangements", "complementary pathways for admission",
                    "registration and documentation support", "livelihood inclusion programmes"],
        "actor": ["host States", "the Office of the High Commissioner for Refugees",
                  "front-line humanitarian responders"],
    },
    "cybersecurity": {
        "issue": ["attacks against critical infrastructure", "the proliferation of intrusion tools",
                  "the targeting of humanitarian data", "ransomware against health systems"],
        "instrument": ["the framework of responsible State behaviour in cyberspace",
                       "the Programme of Action on cyber", "applicable international law"],
        "measure": ["computer emergency response capacity", "confidence-building measures",
                    "national points of contact", "incident attribution procedures"],
        "actor": ["national CERTs", "critical infrastructure operators",
                  "developing States seeking capacity-building"],
    },
    "artificial intelligence governance": {
        "issue": ["the deployment of autonomous weapons systems",
                  "algorithmic discrimination in public services",
                  "the concentration of compute capacity", "synthetic media in electoral periods"],
        "instrument": ["the Universal Declaration of Human Rights",
                       "existing international humanitarian law",
                       "the Global Digital Compact"],
        "measure": ["meaningful human control", "independent audit requirements",
                    "incident reporting mechanisms", "capacity-building for regulators"],
        "actor": ["developing States", "national human rights institutions",
                  "the scientific advisory panel"],
    },
    "food security": {
        "issue": ["disruption of grain corridors", "acute malnutrition among children",
                  "the volatility of fertilizer markets", "post-harvest losses"],
        "instrument": ["the Sustainable Development Goals",
                       "the Committee on World Food Security guidelines",
                       "the Voluntary Guidelines on the Right to Adequate Food"],
        "measure": ["strategic grain reserves", "smallholder credit facilities",
                    "school feeding programmes", "market information systems"],
        "actor": ["smallholder farmers", "the World Food Programme",
                  "net food-importing developing countries"],
    },
    "maritime security": {
        "issue": ["piracy in transit corridors", "illegal, unreported and unregulated fishing",
                  "damage to submarine cables", "unsafe migration by sea"],
        "instrument": ["the United Nations Convention on the Law of the Sea",
                       "the SUA Convention", "regional information-sharing arrangements"],
        "measure": ["joint patrol arrangements", "port State control", "vessel monitoring systems",
                    "prosecution and transfer agreements"],
        "actor": ["coastal States", "regional fisheries management organizations",
                  "seafarers and their representatives"],
    },
    "global health": {
        "issue": ["inequitable access to countermeasures", "the erosion of routine immunization",
                  "antimicrobial resistance", "attacks on health-care facilities"],
        "instrument": ["the International Health Regulations", "the pandemic accord",
                       "the Sustainable Development Goals"],
        "measure": ["regional manufacturing capacity", "genomic surveillance networks",
                    "stockpile pre-positioning", "health workforce retention schemes"],
        "actor": ["the World Health Organization", "national health authorities",
                  "community health workers"],
    },
}

# --- clause grammars ------------------------------------------------------
# Clauses are typed by the complement they take, so a verb is never paired with
# a frame it cannot govern ("Decides to the Secretary-General to report ...").
#   np      -> takes a noun phrase          ("Welcomes the establishment of ...")
#   np_to   -> takes an object + infinitive ("Urges all States to strengthen ...")
#   that    -> takes a that-clause          ("Stresses that any measures ...")
#   inf     -> takes a bare infinitive      ("Decides to remain seized ...")

PREAMBULAR_OPENERS = {
    "np": [
        "Recalling", "Reaffirming", "Noting with concern", "Deeply concerned by",
        "Expressing grave concern at", "Bearing in mind", "Emphasizing", "Welcoming",
        "Acknowledging", "Alarmed by", "Mindful of", "Taking note of", "Underlining",
        "Stressing the importance of", "Convinced of the need to address",
    ],
    "that": [
        "Recognizing that", "Noting that", "Concerned that", "Convinced that",
        "Aware that", "Emphasizing that", "Stressing that", "Recalling that",
    ],
}

PREAMBULAR_FRAMES = {
    "np": [
        "its previous resolutions on {topic}, in particular resolution {res_no}",
        "{instrument} and the obligations arising therefrom",
        "the urgent need to address {issue}",
        "the disproportionate impact of {issue} on {actor}",
        "the report of the Secretary-General on {topic} ({doc_no})",
        "the contribution of {measure} to the protection of {actor}",
        "the primary responsibility of States for the implementation of {instrument}",
        "the outcome of the {ordinal} session of the {committee}",
        "the persistence of {issue} in several regions",
    ],
    "that": [
        "{issue} continues to undermine the objectives of {instrument}",
        "durable solutions to {issue} require sustained international cooperation",
        "{measure} cannot be sustained without predictable financing",
        "the situation of {actor} demands urgent and coordinated action",
        "nothing in the present resolution affects the rights and obligations of States "
        "under {instrument}",
    ],
}

OPERATIVE_VERBS = {
    "np_to": [
        "Calls upon", "Urges", "Requests", "Encourages", "Invites",
        "Reiterates its call upon", "Appeals to",
    ],
    "np": [
        "Welcomes", "Takes note of", "Endorses", "Notes with appreciation",
        "Commends", "Supports",
    ],
    # negative verbs get their own complements so nothing "deplores the report"
    "np_neg": ["Condemns", "Deplores", "Strongly condemns", "Regrets"],
    "that": [
        "Stresses", "Emphasizes", "Reaffirms", "Decides", "Recognizes", "Considers",
        "Notes", "Underlines",
    ],
    "inf": ["Decides to", "Resolves to", "Further decides to"],
}

OPERATIVE_FRAMES = {
    "np_to": [
        "all States to strengthen {measure} in accordance with {instrument}",
        "Member States to allocate predictable and additional resources for {measure}",
        "the Secretary-General to report to the {organ_short} at its {ordinal} session "
        "on the implementation of the present resolution",
        "States parties to refrain from any action that would aggravate {issue}",
        "{actor} to participate fully and effectively in the design of {measure}",
        "international financial institutions to expand concessional support for {measure}",
        "all relevant entities of the United Nations system to mainstream {topic} within "
        "their existing mandates",
        "regional and subregional organizations to share good practices concerning {measure}",
    ],
    "np": [
        "the establishment of a working group, open to all Member States, to elaborate "
        "recommendations on {topic}",
        "the report of the Secretary-General on {topic} ({doc_no})",
        "the efforts of {actor} to implement {instrument} despite limited resources",
        "the contribution of {measure} to the objectives of {instrument}",
    ],
    "np_neg": [
        "all acts that aggravate {issue}, wherever they occur",
        "the persistence of {issue} in defiance of {instrument}",
        "any attempt to obstruct the access of {actor} to {measure}",
        "the diversion of resources away from {measure}",
    ],
    "that": [
        "any measures adopted in response to {issue} shall be consistent with "
        "international law, including {instrument}",
        "technology transfer on mutually agreed terms is essential to {measure}",
        "the needs of {actor} shall be reflected in all follow-up processes",
        "the present resolution shall be implemented without prejudice to {instrument}",
        "capacity-building for {measure} should be demand-driven and sustained",
    ],
    "inf": [
        "remain seized of the matter",
        "convene an open-ended working group on {topic} during its {ordinal} session",
        "include in the provisional agenda of its {ordinal} session an item entitled "
        "“{topic}”",
        "establish a voluntary trust fund in support of {measure}",
    ],
}

DEBATE_OPENERS = [
    "Thank you, Mr. President.", "Thank you, Madam Chair.",
    "I thank the President for convening this meeting.",
    "At the outset, my delegation congratulates you on assuming the chairmanship.",
]

DEBATE_MOVES = [
    "My delegation aligns itself with the statement delivered by the distinguished "
    "representative of {bloc}, and wishes to add the following in its national capacity.",
    "We are gravely concerned by {issue}, which threatens the credibility of {instrument}.",
    "For {country}, {topic} is not an abstract question: it is a matter of survival.",
    "We therefore call for the immediate strengthening of {measure}.",
    "My delegation cannot support language that dilutes the obligations set out in {instrument}.",
    "We would welcome clarification from the sponsors regarding operative paragraph {para_no}.",
    "In a spirit of compromise, we are prepared to accept the amended text, on the "
    "understanding that {measure} remains adequately resourced.",
    "Any solution must place {actor} at the centre of implementation.",
    "We regret that the draft before us does not adequately reflect the concerns of {bloc}.",
    "My delegation reserves its position on operative paragraph {para_no} pending "
    "instructions from our capital.",
]

DEBATE_CLOSERS = [
    "I thank you, Mr. President.", "Thank you for your kind attention.",
    "My delegation stands ready to engage constructively. Thank you.",
]

POINTS = [
    "Point of order: the speaker has exceeded the agreed time limit.",
    "Point of information to the delegate of {country}: how does your proposal address "
    "the financing of {measure}?",
    "Motion to move into a moderated caucus of ten minutes, with a speaking time of "
    "forty-five seconds, on the topic of {topic}.",
    "Motion to divide the question on operative paragraph {para_no}.",
    "Right of reply requested by the delegation of {country}.",
]

CHAIR_LINES = [
    "The Chair recognizes the delegate of {country}.",
    "The motion is in order. We will proceed to a vote by show of placards.",
    "The motion carries. The floor is open for the moderated caucus.",
    "The Chair reminds delegations that amendments must be submitted in writing.",
    "We shall now proceed to take action on draft resolution {res_no}.",
    "The draft resolution is adopted by {yes} votes to {no}, with {abst} abstentions.",
]

ORDINALS = [
    "seventy-fourth", "seventy-fifth", "seventy-sixth", "seventy-seventh",
    "seventy-eighth", "seventy-ninth", "eightieth", "eighty-first",
]

DOC_KINDS = ("resolution", "debate", "position_paper", "procedural")


class _Ctx:
    """Fills clause templates with topic-consistent vocabulary."""

    def __init__(self, rng: random.Random, topic: str):
        self.rng = rng
        self.topic = topic
        bank = TOPICS[topic]
        self.bank = bank
        organ_long, organ_short, prefix = rng.choice(ORGANS)
        self.organ_long, self.organ_short, self.prefix = organ_long, organ_short, prefix
        self.session = rng.choice(ORDINALS)
        self.committee = rng.choice(COMMITTEES)
        self.res_no = f"{prefix}/{rng.randint(70, 81)}/{rng.randint(1, 320)}"
        self.doc_no = f"A/{rng.randint(70, 81)}/{rng.randint(100, 900)}"

    def fill(self, frame: str, **extra) -> str:
        r = self.rng
        values = dict(
            topic=self.topic,
            issue=r.choice(self.bank["issue"]),
            instrument=r.choice(self.bank["instrument"]),
            measure=r.choice(self.bank["measure"]),
            actor=r.choice(self.bank["actor"]),
            country=r.choice(DELEGATIONS),
            bloc=r.choice(BLOCS),
            organ_short=self.organ_short.title(),
            committee=self.committee,
            ordinal=r.choice(ORDINALS),
            res_no=self.res_no,
            doc_no=self.doc_no,
            para_no=r.randint(1, 12),
            yes=r.randint(90, 170), no=r.randint(0, 20), abst=r.randint(0, 40),
        )
        values.update(extra)
        return frame.format(**values)

    def preambular(self) -> str:
        """One preambular clause: opener + a complement it can actually govern."""
        kind = self.rng.choice(["np", "np", "that"])  # NP clauses are the common case
        opener = self.rng.choice(PREAMBULAR_OPENERS[kind])
        body = self.fill(self.rng.choice(PREAMBULAR_FRAMES[kind]))
        return f"{opener} {body},"

    def operative(self) -> str:
        """One operative clause, verb and complement type matched."""
        kind = self.rng.choices(
            ["np_to", "np", "np_neg", "that", "inf"], weights=[5, 3, 1, 3, 2], k=1
        )[0]
        verb = self.rng.choice(OPERATIVE_VERBS[kind])
        body = self.fill(self.rng.choice(OPERATIVE_FRAMES[kind]))
        return f"{verb} that {body}" if kind == "that" else f"{verb} {body}"


def make_resolution(rng: random.Random, topic: str) -> str:
    c = _Ctx(rng, topic)
    lines = [
        f"{c.organ_short}",
        f"Session: {c.session}    Agenda item {rng.randint(9, 148)}",
        f"Document: {c.res_no}",
        f"Title: Resolution on {topic}",
        "",
        f"{c.organ_long},",
        "",
    ]
    for _ in range(rng.randint(3, 7)):
        lines.append(c.preambular())
        lines.append("")
    n_op = rng.randint(4, 9)
    for i in range(1, n_op + 1):
        clause = c.operative()
        lines.append(f"{i}. {clause};" if i < n_op else f"{i}. {clause}.")
        lines.append("")
    abst = rng.randint(0, 45)
    lines.append(f"Adopted at the {rng.randint(2, 70)}th plenary meeting, "
                 f"by {rng.randint(95, 178)} votes to {rng.randint(0, 12)}, "
                 f"with {abst} abstention{'' if abst == 1 else 's'}.")
    return "\n".join(lines)


def make_debate(rng: random.Random, topic: str) -> str:
    c = _Ctx(rng, topic)
    lines = [
        f"{c.organ_short} -- {c.committee}",
        f"Verbatim record, {c.session} session. Agenda: {topic}.",
        "",
    ]
    for _ in range(rng.randint(3, 6)):
        country = rng.choice(DELEGATIONS)
        lines.append(c.fill(rng.choice(CHAIR_LINES), country=country))
        lines.append(f"DELEGATE OF {country.upper()}:")
        speech = [rng.choice(DEBATE_OPENERS)]
        for frame in rng.sample(DEBATE_MOVES, rng.randint(2, 5)):
            speech.append(c.fill(frame, country=country))
        speech.append(rng.choice(DEBATE_CLOSERS))
        lines.append(" ".join(speech))
        lines.append("")
        if rng.random() < 0.45:
            lines.append(c.fill(rng.choice(POINTS), country=rng.choice(DELEGATIONS)))
            lines.append("")
    return "\n".join(lines)


def make_position_paper(rng: random.Random, topic: str) -> str:
    c = _Ctx(rng, topic)
    country = rng.choice(DELEGATIONS)
    lines = [
        "POSITION PAPER",
        f"Delegation: {country}",
        f"Committee: {c.committee}",
        f"Topic: {topic}",
        "",
        "I. Background",
        c.fill("The delegation of {country} notes with concern that {issue} continues to "
               "impede the full implementation of {instrument}. Successive reports of the "
               "Secretary-General ({doc_no}) have documented the effect of this situation "
               "on {actor}.", country=country),
        "",
        "II. National Position",
        c.fill("{country} holds that any credible response must combine {measure} with "
               "predictable financing and clear reporting obligations. We recall the "
               "commitments undertaken under {instrument} and stress that they are not "
               "subject to renegotiation.", country=country),
        "",
        "III. Proposed Solutions",
    ]
    for i in range(1, 4):
        lines.append(f"{i}. {c.operative()}.")
    lines += [
        "",
        "IV. Partners",
        c.fill("{country} is prepared to co-sponsor a draft resolution with {bloc} and "
               "welcomes consultations with all interested delegations.", country=country),
    ]
    return "\n".join(lines)


def make_procedural(rng: random.Random, topic: str) -> str:
    c = _Ctx(rng, topic)
    lines = [f"PROCEDURAL RECORD -- {c.committee}", f"Topic under consideration: {topic}", ""]
    for _ in range(rng.randint(4, 9)):
        country = rng.choice(DELEGATIONS)
        lines.append(c.fill(rng.choice(POINTS), country=country))
        lines.append(c.fill(rng.choice(CHAIR_LINES), country=rng.choice(DELEGATIONS)))
        lines.append("")
    return "\n".join(lines)


MAKERS = {
    "resolution": make_resolution,
    "debate": make_debate,
    "position_paper": make_position_paper,
    "procedural": make_procedural,
}

#: sampling weights -- resolutions are the primary drafting target
WEIGHTS = {"resolution": 0.45, "debate": 0.30, "position_paper": 0.15, "procedural": 0.10}

DOC_SEPARATOR = "\n\n<|endoftext|>\n\n"


def build_corpus(n_docs: int = 4000, seed: int = 7, topics: list[str] | None = None) -> str:
    """Generate ``n_docs`` documents as one training-ready string."""
    rng = random.Random(seed)
    topics = topics or list(TOPICS)
    kinds = list(WEIGHTS)
    weights = [WEIGHTS[k] for k in kinds]
    docs = []
    for _ in range(n_docs):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        docs.append(MAKERS[kind](rng, rng.choice(topics)))
    return DOC_SEPARATOR.join(docs) + "\n"


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate the synthetic diplomacy corpus")
    p.add_argument("--out", default="data/raw/diplomacy.txt")
    p.add_argument("--docs", type=int, default=4000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    text = build_corpus(a.docs, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"{a.docs} documents | {len(text):,} characters -> {out}")


if __name__ == "__main__":
    _cli()
