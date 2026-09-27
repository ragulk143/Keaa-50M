"""
pull_data.py — Collects public GitHub Actions YAML, Dockerfiles, and Bash scripts
for training Keaa-50M, a small Mamba SSM autocomplete model for infra-as-code.

Requires: requests   (pip install requests)
Set the GITHUB_TOKEN environment variable to a personal access token
(no special scopes needed for public code search) to avoid GitHub's very
low unauthenticated rate limits (10 req/min vs ~30 req/min authenticated,
and search-specific limits on top of that).
"""

import os
import time
import json
import base64
import pathlib
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
API_ROOT = "https://api.github.com"
OUT_DIR = pathlib.Path("raw_data")

# (search query, output subdir)
TARGETS = [
    ("path:.github/workflows extension:yml", "workflows"),
    ("path:.github/workflows extension:yaml", "workflows"),
    ("filename:Dockerfile", "dockerfiles"),
    ("extension:sh", "bash"),
]

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def search_code(query, max_results=200):
    """Yield GitHub code-search results for a query, paging until max_results."""
    results = []
    page = 1
    while len(results) < max_results:
        resp = requests.get(
            f"{API_ROOT}/search/code",
            headers=HEADERS,
            params={"q": query, "per_page": 100, "page": page},
        )
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 5)
            print(f"Rate limited. Sleeping {wait:.0f}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        results.extend(items)
        page += 1
        if page > 10:  # GitHub code search caps at 1000 results (10 x 100)
            break
        time.sleep(2)  # stay under the search-specific rate limit
    return results[:max_results]


def fetch_file_content(item):
    """Download raw content for a single search-result item."""
    resp = requests.get(item["url"], headers=HEADERS)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("encoding") == "base64":
        try:
            return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
        except Exception:
            return None
    return None


def main(max_per_target=200):
    if not GITHUB_TOKEN:
        print("WARNING: no GITHUB_TOKEN set — you will hit rate limits fast.")

    OUT_DIR.mkdir(exist_ok=True)
    manifest = []

    for query, subdir in TARGETS:
        print(f"Searching: {query}")
        items = search_code(query, max_results=max_per_target)
        target_dir = OUT_DIR / subdir
        target_dir.mkdir(exist_ok=True)

        for i, item in enumerate(items):
            content = fetch_file_content(item)
            if not content or len(content) < 20:
                continue
            fname = target_dir / f"{item['repository']['full_name'].replace('/', '__')}__{i}.txt"
            fname.write_text(content, encoding="utf-8")
            manifest.append({
                "repo": item["repository"]["full_name"],
                "path": item["path"],
                "local_file": str(fname),
                "category": subdir,
            })
            time.sleep(0.5)  # be polite to the contents API too

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Collected {len(manifest)} files across {len(TARGETS)} targets.")
    print(f"Manifest saved to {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
