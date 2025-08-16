from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


def rerank(query: str, docs: List[object], model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> List[Tuple[float, object]]:
    """Return list of (score, doc) sorted desc. If CrossEncoder not available, return original docs with score 0."""
    if CrossEncoder is None:
        return [(0.0, d) for d in docs]

    model = CrossEncoder(model_name)
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    scored = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
