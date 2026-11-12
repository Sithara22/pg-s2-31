"""Extract simple topic labels and keywords from learner questions."""

import re


# Extend this vocabulary as your construction topics grow.
TOPIC_PATTERNS = {
    "roofing": r"\b(?:roof|roofs|roofing|truss|trusses|rafter|rafters)\b",
    "waterproofing": r"\b(?:waterproof|waterproofing|membrane|membranes|leak|leaks)\b",
    "wall framing": r"\b(?:wall frame|wall frames|wall framing|stud|studs|bracing)\b",
    "concrete": r"\b(?:concrete|slab|slabs|cement|curing)\b",
    "foundations": r"\b(?:foundation|foundations|footing|footings)\b",
    "plumbing": r"\b(?:plumbing|pipe|pipes|drain|drains|drainage)\b",
    "electrical": r"\b(?:electrical|wiring|circuit|circuits|switchboard)\b",
    "insulation": r"\b(?:insulation|insulate|thermal|acoustic)\b",
    "fire safety": r"\b(?:fire|fireproof|fireproofing|combustible)\b",
    "site safety": r"\b(?:ppe|hazard|hazards|scaffold|scaffolding|fall protection)\b",
}

STOP_WORDS = set(
    """
    a an the and or but if then than to of in on at by for from with
    without into is are was were be been being do does did can could
    should would will may might must i me my we our you your they
    their it its this that these those what which who when where why
    how please tell explain about need know not no
    """.split()
)


def classify_question(question: str) -> tuple[list[str], list[str]]:
    text = re.sub(r"\s+", " ", question.casefold()).strip()

    topics = sorted(
        topic
        for topic, pattern in TOPIC_PATTERNS.items()
        if re.search(pattern, text)
    )

    tokens = re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", text)

    # Preserve first appearance while removing duplicate keywords.
    keywords = list(
        dict.fromkeys(
            token
            for token in tokens
            if len(token) > 2 and token not in STOP_WORDS
        )
    )[:15]

    return topics or ["unclassified"], keywords