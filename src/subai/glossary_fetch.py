"""Refresh series metadata from the official TMDB (v3) and TVDB (v4) APIs.

Writes two NEW files next to the curated glossary and never modifies curated files:
  fetched.yaml  - titles/aliases from both sources, cast, and a diff against characters.yaml
  episodes.yaml - per episode: titles + overviews (tr, en), air date, guest stars (scene context)

This is the only code that uses the network, and only when you run `subai glossary-fetch`.
Keys come from the environment (TMDB_API_KEY, TVDB_API_KEY), never from files in the repo.
"""
import json
import logging
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
TMDB = "https://api.themoviedb.org/3"
TVDB = "https://api4.thetvdb.com/v4"


class FetchError(Exception):
    pass


def _http(url: str, headers: dict | None = None, body: bytes | None = None, retries: int = 3) -> dict:
    where = url.split("?")[0]  # never put the query string (api_key) in messages
    for i in range(retries):
        req = urllib.request.Request(url, headers=headers or {}, data=body)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(2 * (i + 1))
                continue
            raise FetchError(f"HTTP {exc.code} from {where}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if i < retries - 1:
                time.sleep(2 * (i + 1))
                continue
            raise FetchError(f"network error for {where}: {exc.__class__.__name__}") from None
    raise FetchError(f"failed: {where}")


def _tmdb(path: str, key: str, **params) -> dict:
    return _http(f"{TMDB}{path}?" + urllib.parse.urlencode({"api_key": key, **params}))


def norm(s: str) -> str:
    """Accent/case-insensitive key: 'Başol' == 'Basol', 'Çıtanak' == 'citanak'."""
    s = (s or "").replace("ı", "i").replace("İ", "I")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.casefold().split())


def fetch_tmdb(tmdb_id: int, key: str) -> dict:
    en = _tmdb(f"/tv/{tmdb_id}", key, language="en-US")
    tr = _tmdb(f"/tv/{tmdb_id}", key, language="tr-TR")
    alt = _tmdb(f"/tv/{tmdb_id}/alternative_titles", key)
    agg = _tmdb(f"/tv/{tmdb_id}/aggregate_credits", key, language="tr-TR")
    cast = [
        {"actor": c["name"], "character": (r.get("character") or "").strip(), "episodes": r.get("episode_count", 0)}
        for c in agg.get("cast", []) for r in c.get("roles", [])
    ]
    episodes = []
    for s in en.get("seasons", []):
        n = s["season_number"]
        if n < 1:
            continue
        s_en = _tmdb(f"/tv/{tmdb_id}/season/{n}", key, language="en-US")
        s_tr = {e["episode_number"]: e for e in _tmdb(f"/tv/{tmdb_id}/season/{n}", key, language="tr-TR").get("episodes", [])}
        for e in s_en.get("episodes", []):
            t = s_tr.get(e["episode_number"], {})
            episodes.append({
                "season": n, "episode": e["episode_number"], "air_date": e.get("air_date"),
                "runtime_min": e.get("runtime"),
                "title_tr": t.get("name") or None, "title_en": e.get("name") or None,
                "overview_tr": t.get("overview") or None, "overview_en": e.get("overview") or None,
                "guest_stars": [{"actor": g["name"], "character": (g.get("character") or "").strip()}
                                for g in e.get("guest_stars", [])],
            })
    return {
        "details": {"name_en": en.get("name"), "name_tr": tr.get("name"), "original_name": en.get("original_name"),
                    "first_air_date": en.get("first_air_date"), "last_air_date": en.get("last_air_date"),
                    "status": en.get("status"), "episodes": en.get("number_of_episodes"),
                    "seasons": en.get("number_of_seasons"),
                    "networks": [n["name"] for n in en.get("networks", [])]},
        "alternative_titles": [{"country": a["iso_3166_1"], "title": a["title"]} for a in alt.get("results", [])],
        "cast": cast,
        "episodes": episodes,
    }


def fetch_tvdb(tvdb_id: int, key: str) -> dict:
    login = _http(f"{TVDB}/login", {"Content-Type": "application/json"}, json.dumps({"apikey": key}).encode())
    token = (login.get("data") or {}).get("token")
    if not token:
        raise FetchError("TVDB login returned no token (check TVDB_API_KEY)")
    ext = _http(f"{TVDB}/series/{tvdb_id}/extended?short=false", {"Authorization": f"Bearer {token}"})["data"]
    return {
        "name": ext.get("name"), "status": (ext.get("status") or {}).get("name"),
        "first_aired": ext.get("firstAired"),
        "aliases": [{"language": a["language"], "name": a["name"]} for a in ext.get("aliases") or []],
        "name_translations": ext.get("nameTranslations") or [],
        "characters": [{"character": c.get("name"), "actor": c.get("personName")} for c in ext.get("characters") or []],
    }


def diff_characters(curated: list[dict], tmdb_cast: list[dict], tvdb_chars: list[dict]) -> dict:
    """Compare API cast against characters.yaml by accent-insensitive actor name."""
    def key(s: str) -> tuple:  # first + last name, so "Ahmet Somers" == "Ahmet Mark Somers"
        t = norm(s).split()
        return (t[0], t[-1]) if t else ()

    by_actor = {key(c["actor"]): c for c in curated}
    missing, name_diff, count_diff = [], [], []
    for source, rows in (("tmdb", tmdb_cast), ("tvdb", tvdb_chars)):
        for r in rows:
            cur = by_actor.get(key(r["actor"]))
            if cur is None:
                missing.append({"source": source, **{k: r.get(k) for k in ("actor", "character", "episodes")}})
                continue
            ch = norm(r.get("character") or "")
            if ch and not (ch in norm(cur["name"]) or norm(cur["name"]) in ch
                           or any(norm(a) in ch for a in cur.get("aliases", []))):
                name_diff.append({"source": source, "actor": cur["actor"], "curated": cur["name"], "api": r["character"]})
            if source == "tmdb" and r.get("episodes") and cur.get("episodes") and r["episodes"] != cur["episodes"]:
                count_diff.append({"actor": cur["actor"], "curated": cur["episodes"], "tmdb": r["episodes"]})
    return {"missing_in_glossary": missing, "name_differs": name_diff, "episode_count_differs": count_diff}


def run_fetch(series_dir: Path, tmdb_key: str, tvdb_key: str, write: bool) -> dict:
    meta = yaml.safe_load((series_dir / "series.yaml").read_text(encoding="utf-8"))
    curated = yaml.safe_load((series_dir / "characters.yaml").read_text(encoding="utf-8"))["characters"]
    ids = meta["ids"]
    log.info("Fetching TMDB %s ...", ids["tmdb"])
    tmdb = fetch_tmdb(int(ids["tmdb"]), tmdb_key)
    log.info("Fetching TVDB %s ...", ids["tvdb"])
    tvdb = fetch_tvdb(int(ids["tvdb"]), tvdb_key)
    diff = diff_characters(curated, tmdb["cast"], tvdb["characters"])
    fetched = {"retrieved": date.today().isoformat(), "tmdb": {k: v for k, v in tmdb.items() if k != "episodes"},
               "tvdb": tvdb, "diff_vs_characters_yaml": diff}
    episodes = {"retrieved": fetched["retrieved"], "source": f"tmdb:{ids['tmdb']}", "episodes": tmdb["episodes"]}
    if write:
        dump = dict(allow_unicode=True, sort_keys=False, width=100)
        (series_dir / "fetched.yaml").write_text(
            "# Generated by `subai glossary-fetch`. Do not edit; curated data lives in series/characters/terms.yaml.\n"
            + yaml.safe_dump(fetched, **dump), encoding="utf-8")
        (series_dir / "episodes.yaml").write_text(
            "# Generated by `subai glossary-fetch` (TMDB). Scene context for translation; guest stars for ASR priming.\n"
            + yaml.safe_dump(episodes, **dump), encoding="utf-8")
    return {"fetched": fetched, "episodes": episodes}
