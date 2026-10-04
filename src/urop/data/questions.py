"""Canonical question parsing and documented transcript wrapper normalization."""
from pathlib import Path
import re

def normalize_question(text: str) -> str:
    """Remove only the documented machine utterance-ID wrapper."""
    match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s+\((.*)\)", text.strip())
    return (match.group(1) if match else text).strip()

def normalize(text: str) -> str:
    text = normalize_question(text).lower().replace("’", "'").replace("l_a", "la")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())

def question_table(html_path: Path) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_path.read_text(), "html.parser")
    candidates = [table for table in soup.find_all("table")
                  if "how has seeing a therapist affected you" in table.get_text(" ", strip=True).lower()]
    if len(candidates) != 1:
        raise ValueError("Expected one 85-question table from paper Appendix A")
    result = []
    for row in candidates[0].find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) == 3 and re.fullmatch(r"\(\d+\)", cells[0]):
            result.append({"slot": int(cells[0][1:-1]) - 1, "question": cells[1], "type": cells[2]})
    if len(result) != 85 or sorted(r["slot"] for r in result) != list(range(85)):
        raise ValueError("Paper question list is not exactly 85 slots")
    return result
