#!/usr/bin/env python3
import argparse
import ast
import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from dateparser.search import search_dates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DAY_ORDER = ["lunedi", "martedi", "mercoledi", "giovedi", "venerdi", "sabato", "domenica"]
DAY_PATTERN = r"(?:lunedi|martedi|mercoledi|giovedi|venerdi|sabato|domenica)"


CATEGORY_KEYWORDS = {
    "art_exhibitions": [
        "mostra",
        "esposizione",
        "galleria",
        "museo",
        "installazione",
        "arte contemporanea",
        "vernissage",
    ],
    "music_concerts": [
        "concerto",
        "dal vivo",
        "dj set",
        "festival musicale",
        "orchestra",
        "recital",
        "band",
    ],
    "sport_fitness": [
        "corsa",
        "gara",
        "torneo",
        "partita/gara",
        "yoga",
        "fitness",
        "allenamento",
        "ciclismo",
        "maratona",
    ],
    "theatre_shows": [
        "teatro",
        "spettacolo teatrale",
        "commedia",
        "musical",
        "opera",
        "performance teatrale",
        "cinema",
    ],
    "community_lifestyle": [
        "mercato",
        "fiera",
        "festival locale",
        "festa di quartiere",
        "workshop",
        "lab",
        "incontro",
        "conferenza",
        "talk",
        "attività per famiglie/bambini",
        "cibo e bevande",
        "comunità",
    ],
}



def normalize_text(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_title(text: str) -> str:
    return normalize_text(normalize_whitespace(text).lower())

# remove adjaced duplicates
def normalize_location(text: str) -> str:
    words = text.split()
    deduped: List[str] = []
    for word in words:
        if not deduped or deduped[-1].lower() != word.lower():
            deduped.append(word)
    return " ".join(deduped)


def parse_description(raw: str) -> str:
    #Removes leading and trailing spaces/newlines
    raw = raw.strip()
    if not raw:
        return ""
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return normalize_whitespace(raw)
    if isinstance(value, list):
        return normalize_whitespace(" ".join(str(item) for item in value))
    return normalize_whitespace(str(value))

# NORMALIZE PRICE

# finds zero price 
FREE_PRICE_RE = re.compile(r"\b0\s*euro\b|\b0[.,]00\b", re.IGNORECASE)


def normalize_price(raw: str) -> str:
    cleaned = normalize_whitespace(raw)
    #empty case
    if not cleaned:
        return "Non disponibile"
    lowered = normalize_text(cleaned.lower())
    # non disponibile found in string
    if (
        "non disponibile" in lowered
        or "non disp" in lowered
        or lowered in {"nd", "n/d", "na", "n.a", "non specificato", "non specificata"}
    ):
        return "Non disponibile"
    
    # gratis found in string
    if (
        "gratis" in lowered
        or "gratuit" in lowered
        or "ingresso libero" in lowered
        or "ingresso gratuito" in lowered
        or "free" in lowered
        or FREE_PRICE_RE.search(lowered)
    ):
        return "Gratuito"
    return "A pagamento"

# NORMALIZE DATE TIME

# Find range with words separators
TIME_RANGE_DALLE_RE = re.compile(
    r"(?:dalle|da)\s*(\d{1,2})(?:[:.](\d{1,2}))?\s*(?:alle|al|a)\s*(\d{1,2})(?:[:.](\d{1,2}))?",
    re.IGNORECASE,
)
# Find range with dash separator
TIME_RANGE_DASH_RE = re.compile(
    r"(\d{1,2})(?:[:.](\d{1,2}))?\s*[-\u2013]\s*(\d{1,2})(?:[:.](\d{1,2}))?",
    re.IGNORECASE,
)
# Find time
TIME_TOKEN_RE = re.compile(r"\b(\d{1,2})(?:[:.](\d{1,2}))?\b")


def normalize_minute(minute: Optional[str]) -> int:
    if not minute:
        return 0
    if len(minute) == 1:
        return int(minute) * 10
    return int(minute)


def format_time(hour: str, minute: Optional[str]) -> str:
    return f"{int(hour):02d}:{normalize_minute(minute):02d}"


def extract_time_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    lowered = text.lower()
    # when it's closed
    if re.search(r"\bchius", lowered):
        return None, None

    cleaned = re.sub(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", " ", lowered)

    range_match = TIME_RANGE_DALLE_RE.search(cleaned)
    if range_match:
        return (
            format_time(range_match.group(1), range_match.group(2)),
            format_time(range_match.group(3), range_match.group(4)),
        )

    range_match = TIME_RANGE_DASH_RE.search(cleaned)
    if range_match:
        return (
            format_time(range_match.group(1), range_match.group(2)),
            format_time(range_match.group(3), range_match.group(4)),
        )

    single_match = TIME_TOKEN_RE.search(cleaned)
    if single_match:
        return format_time(single_match.group(1), single_match.group(2)), None

    return None, None


DAY_GROUP_RE = re.compile(
    rf"(?P<days>{DAY_PATTERN}(?:\s*[-\u2013]\s*{DAY_PATTERN})?(?:\s*(?:,|e)\s*{DAY_PATTERN})*)"
    rf"\s*(?P<rest>.*?)(?=(?:{DAY_PATTERN})\b|$)",
    re.IGNORECASE,
)


def expand_day_group(day_group: str) -> List[str]:
    day_group = normalize_text(day_group.lower())
    if "-" in day_group or "\u2013" in day_group:
        sep = "-" if "-" in day_group else "\u2013"
        left, right = [part.strip() for part in day_group.split(sep, 1)]
        left = re.split(r"[\s,]", left)[0]
        right = re.split(r"[\s,]", right)[0]
        if left in DAY_ORDER and right in DAY_ORDER:
            start = DAY_ORDER.index(left)
            end = DAY_ORDER.index(right)
            if start <= end:
                return DAY_ORDER[start : end + 1]
            return DAY_ORDER[start:] + DAY_ORDER[: end + 1]
    days = re.split(r"\s*(?:,|e)\s*", day_group)
    return [day for day in days if day in DAY_ORDER]


def extract_day_time_map(
    text: str,
) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]], Optional[str], Optional[str]]:
    lowered = normalize_text(text.lower())
    if "tutti i giorni" in lowered:
        default_open, default_close = extract_time_range(lowered)
        return {}, {}, default_open, default_close

    matches = list(DAY_GROUP_RE.finditer(lowered))
    if not matches:
        default_open, default_close = extract_time_range(lowered)
        return {}, {}, default_open, default_close

    day_open: Dict[str, Optional[str]] = {}
    day_close: Dict[str, Optional[str]] = {}
    for match in matches:
        days = expand_day_group(match.group("days"))
        rest = match.group("rest")
        open_time, close_time = extract_time_range(rest)
        for day in days:
            day_open[day] = open_time
            day_close[day] = close_time
    return day_open, day_close, None, None


DATE_YEAR_RE = re.compile(r"\b\d{4}\b")
DATE_YEAR_SHORT_RE = re.compile(r"[./-]\d{2}\b")


def has_explicit_year(text: str) -> bool:
    return bool(DATE_YEAR_RE.search(text) or DATE_YEAR_SHORT_RE.search(text))


def parse_date_candidates(text: str, default_year: int) -> List[date]:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\b\d{1,2}[:.]\d{2}\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}\s*[-\u2013]\s*\d{1,2}\b", " ", cleaned)
    # dateparser handles multiple formats and languages; we normalize missing years below.
    results = search_dates(
        cleaned,
        languages=["it", "en"],
        settings={
            "DATE_ORDER": "DMY",
            "PREFER_DAY_OF_MONTH": "first",
        },
    )
    if not results:
        return []
    parsed_dates: List[date] = []
    for match_text, dt in results:
        if dt is None:
            continue
        if not re.search(r"\d", match_text):
            continue
        if not has_explicit_year(match_text):
            try:
                dt = dt.replace(year=default_year)
            except ValueError:
                # Invalid date (e.g., Feb 29 in a non-leap year) -> skip.
                continue
        parsed_dates.append(dt.date())
    return parsed_dates


def parse_date_range(text: str, default_year: int) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    dates = parse_date_candidates(text, default_year)
    if not dates:
        return None, None
    start = min(dates)
    end = max(dates)
    return start.isoformat(), end.isoformat()


def iter_dates(start: str, end: str) -> List[date]:
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    days: List[date] = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def build_dates_list(
    start: Optional[str],
    end: Optional[str],
    day_open: Dict[str, Optional[str]],
    day_close: Dict[str, Optional[str]],
    default_open: Optional[str],
    default_close: Optional[str],
) -> List[Dict[str, Optional[str]]]:
    if not start or not end:
        return []
    dates_list: List[Dict[str, Optional[str]]] = []
    for current in iter_dates(start, end):
        weekday = DAY_ORDER[current.weekday()]
        opening = day_open.get(weekday, default_open)
        closing = day_close.get(weekday, default_close)
        dates_list.append(
            {
                "date": current.isoformat(),
                "weekday": weekday,
                "opening_hour": opening,
                "closing_hour": closing,
            }
        )
    return dates_list


def parse_blocks(text: str) -> List[Dict[str, str]]:
    blocks = [block for block in text.split("\n---\n") if block.strip()]
    entries: List[Dict[str, str]] = []
    for block in blocks:
        entry = {
            "title": "",
            "location": "",
            "date_raw": "",
            "price": "",
            "description": "",
            "url": "",
        }
        for line in block.splitlines():
            if line.startswith("Titolo:"):
                entry["title"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("url:"):
                entry["url"] = line.split(":", 1)[1].strip()
            elif line.startswith("Indirizzo:"):
                entry["location"] = line.split(":", 1)[1].strip()
            elif line.startswith("Data_ora:"):
                entry["date_raw"] = line.split(":", 1)[1].strip()
            elif line.startswith("Prezzo:"):
                entry["price"] = line.split(":", 1)[1].strip()
            elif line.startswith("Descrizione:"):
                entry["description"] = line.split(":", 1)[1].strip()
        entries.append(entry)
    return entries


def coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def parse_json_entries(data: object) -> List[Dict[str, str]]:
    if isinstance(data, list):
        items: Iterable[object] = data
    elif isinstance(data, dict):
        if isinstance(data.get("events"), list):
            items = data["events"]
        else:
            items = [data]
    else:
        return []

    entries: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entry = {
            "title": coerce_text(item.get("titolo") or item.get("title")).strip(),
            "location": coerce_text(item.get("location") or item.get("indirizzo")).strip(),
            "date_raw": coerce_text(item.get("data_ora") or item.get("date_raw") or item.get("date")).strip(),
            "price": coerce_text(item.get("prezzo") or item.get("price")).strip(),
            "description": coerce_text(item.get("event_info") or item.get("description")).strip(),
            "url": coerce_text(item.get("url") or item.get("link")).strip(),
        }
        entries.append(entry)
    return entries


def build_category_model(descriptions: List[str]) -> Tuple[TfidfVectorizer, List[str], List[str]]:
    labels = list(CATEGORY_KEYWORDS.keys())
    prototype_texts = [" ".join(CATEGORY_KEYWORDS[label]) for label in labels]

    vectorizer = TfidfVectorizer(
        preprocessor=lambda txt: normalize_text(txt.lower()),
        ngram_range=(1, 2),
        min_df=1,
    )
    vectorizer.fit(prototype_texts + descriptions)
    return vectorizer, labels, prototype_texts


def assign_category(
    description: str,
    vectorizer: TfidfVectorizer,
    labels: List[str],
    prototype_texts: List[str],
) -> str:
    matrix = vectorizer.transform(prototype_texts + [description])
    proto_vectors = matrix[:-1]
    doc_vector = matrix[-1]
    sims = cosine_similarity(doc_vector, proto_vectors).flatten()
    if not sims.size or sims.max() == 0:
        return "community_lifestyle"
    return labels[int(sims.argmax())]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse events information + TF-IDF categories."
    )
    parser.add_argument(
        "--input",
        default="raw_events.json",
        help="Path to input file (text blocks or JSON).",
    )
    parser.add_argument(
        "--output",
        default="events_tfidf_processed.json",
        help="Path to the JSON output file.",
    )
    parser.add_argument(
        "--default-year",
        type=int,
        default=2023,
        help="Fallback year when missing in date strings.",
    )
    parser.add_argument(
        "--evaluated-url",
        default=None,
        help="Path to newline-delimited URLs to keep.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".json":
        entries = parse_json_entries(json.loads(input_path.read_text(encoding="utf-8")))
    else:
        input_text = input_path.read_text(encoding="utf-8")
        entries = parse_blocks(input_text)

    if args.evaluated_url:
        url_filter = {
            line.strip()
            for line in Path(args.evaluated_url).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        filtered_entries = []
        for entry in entries:
            title_value = normalize_whitespace(entry["title"])
            url_value = entry.get("url")
            if url_value and url_value in url_filter:
                entry["url"] = url_value
                filtered_entries.append(entry)
        entries = filtered_entries

    descriptions: List[str] = []
    for entry in entries:
        desc = parse_description(entry["description"])
        text = desc or entry["title"]
        descriptions.append(text)

    vectorizer, labels, prototype_texts = build_category_model(descriptions)

    output = []
    for entry, desc in zip(entries, descriptions):
        date_raw = entry["date_raw"]
        start_date, end_date = parse_date_range(date_raw, args.default_year)
        day_open, day_close, default_open, default_close = extract_day_time_map(date_raw)
        dates_list = build_dates_list(start_date, end_date, day_open, day_close, default_open, default_close)
        title_value = normalize_whitespace(entry["title"])
        url_value = entry.get("url")

        output.append(
            {
                "title": title_value,
                "location": normalize_location(normalize_whitespace(entry["location"])) or "N/A",
                "price": normalize_price(entry["price"]),
                "category": assign_category(desc, vectorizer, labels, prototype_texts),
                "dates": dates_list,
                "description": desc,
                "url": url_value,
            }
        )

    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
