import json
import re
import time
import hashlib
import random
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup


SITEMAP_URL = "https://www.sundhed.dk/sitemap-laegehaandbog.xml"

MAX_PAGES = 0
TIMEOUT = 30

OUTPUT_SUBFOLDER = "sundhed"
ALLOW_PATH_PREFIX = "/sundhedsfaglig/laegehaandbogen/"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def output_dir() -> Path:
    d = project_root() / "data" / "scrape" / OUTPUT_SUBFOLDER
    d.mkdir(parents=True, exist_ok=True)
    return d


def polite_sleep(last_request_ts: float) -> float:
    delay = random.uniform(27, 37)
    now = time.time()
    wait = (last_request_ts + delay) - now
    if wait > 0:
        time.sleep(wait)
    return time.time()


def fetch(session: requests.Session, url: str) -> requests.Response:
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def parse_sitemap_urls(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    urls = []
    for url_el in root.findall(".//{*}url"):
        loc = url_el.find("{*}loc")
        if loc is not None and loc.text:
            urls.append(loc.text.strip())
    return urls


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def soup_text(el):
    return clean_text(el.get_text(" ", strip=True)) if el else ""


def extract_sections(html: str):
    soup = BeautifulSoup(html, "html.parser")

    title = soup_text(soup.find("h1"))
    meta_desc = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_desc = clean_text(md["content"])

    main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup.find(
        attrs={"role": "document"}
    ) or soup.body or soup

    for tag in main.find_all(["nav", "header", "footer", "aside", "script", "style", "noscript"]):
        tag.decompose()

    sections = []
    current = {"heading": None, "content": []}

    for el in main.find_all(["h2", "h3", "p", "ul", "ol"], recursive=True):
        if el.name in ["h2", "h3"]:
            if current["heading"] is not None or current["content"]:
                current["content"] = [c for c in current["content"] if c != "" and c != []]
                sections.append(current)
            current = {"heading": soup_text(el), "content": []}

        elif el.name == "p":
            txt = soup_text(el)
            if txt:
                current["content"].append(txt)

        elif el.name in ["ul", "ol"]:
            items = [clean_text(li.get_text(" ", strip=True)) for li in el.find_all("li", recursive=False)]
            items = [i for i in items if i]
            if items:
                current["content"].append(items)

    if current["heading"] is not None or current["content"]:
        current["content"] = [c for c in current["content"] if c != "" and c != []]
        sections.append(current)

    return title, meta_desc, sections


def url_to_filename(url: str) -> str:
    p = urlparse(url)
    base = p.path.strip("/").replace("/", "__")
    if not base:
        base = "index"
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{base}__{h}.json"


def is_allowed(url: str) -> bool:
    p = urlparse(url)

    if p.netloc != "www.sundhed.dk":
        return False

    if not p.path.startswith(ALLOW_PATH_PREFIX):
        return False

    if "/illustrationer/" in p.path:
        return False

    return True


def existing_scraped_urls(out: Path) -> set[str]:
    urls = set()
    for fp in out.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            u = (data or {}).get("url")
            if isinstance(u, str) and u.strip():
                urls.add(u.strip())
        except Exception:
            continue
    return urls


def atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run():
    out = output_dir()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; sundhed-scraper/1.0)",
            "Accept-Language": "da,en;q=0.8",
        }
    )

    scraped_urls = existing_scraped_urls(out)

    print("Fetching sitemap...")
    sitemap_xml = fetch(session, SITEMAP_URL).text
    urls = parse_sitemap_urls(sitemap_xml)
    urls = [u for u in urls if is_allowed(u)]

    total_in_scope = len(urls)

    missing = [u for u in urls if u not in scraped_urls]

    if MAX_PAGES:
        missing = missing[:MAX_PAGES]

    print(f"In-scope URLs (prefix {ALLOW_PATH_PREFIX}): {total_in_scope}")
    print(f"Already scraped URLs (from existing JSON files): {len(scraped_urls)}")
    print(f"Missing (will process): {len(missing)} (cap={MAX_PAGES or 'none'})")
    print("Throttle: random delay")
    print(f"Saving to: {out}")

    last_ts = 0.0
    new_scrapes = 0
    errors = 0

    for i, url in enumerate(missing, 1):
        fp = out / url_to_filename(url)

        try:
            last_ts = polite_sleep(last_ts)
            r = fetch(session, url)

            title, meta_desc, sections = extract_sections(r.text)
            data = {
                "url": url,
                "title": title,
                "meta_description": meta_desc,
                "sections": sections,
                "scraped_at": datetime.now().isoformat(),
                "status_code": r.status_code,
            }

            atomic_write_json(fp, data)
            new_scrapes += 1

            print(f"[{i}/{len(missing)}] saved: {fp.name}")

        except Exception as e:
            errors += 1
            print(f"[{i}/{len(missing)}] ERROR {url}: {e}")

    print("-----")
    print(f"Finished. New: {new_scrapes}, Errors: {errors}, Total missing at start: {len(missing)}")


if __name__ == "__main__":
    run()