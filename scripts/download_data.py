"""Download WAY-EEG-GAL participant archives from the official figshare collection.

Collection: https://doi.org/10.6084/m9.figshare.988376 (Luciw, Jarocka & Edin 2014).
Files are fetched from the original source and never redistributed. Partial downloads
resume via HTTP Range requests; every archive is verified against figshare's MD5.

Usage: python scripts/download_data.py [--out data/raw] [--participants 1 2 ...] [--no-extract]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.figshare.com/v2"
COLLECTION_ID = 988376
CHUNK = 1 << 20


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def list_participant_files() -> dict[int, dict]:
    """Return {participant_number: file_record} for the 12 participant archives."""
    arts = _get_json(f"{API}/collections/{COLLECTION_ID}/articles?page_size=100")
    out: dict[int, dict] = {}
    for a in arts:
        title = a["title"]
        if not title.startswith("Participant"):
            continue
        n = int(title.split()[1])
        files = _get_json(f"{API}/articles/{a['id']}")["files"]
        assert len(files) == 1, f"expected one archive for P{n}, got {len(files)}"
        out[n] = files[0]
    assert sorted(out) == list(range(1, 13)), f"expected participants 1..12, got {sorted(out)}"
    return out


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, retries: int = 20) -> None:
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= size:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab") as f:
                if have and r.status != 206:  # server ignored Range: restart
                    f.truncate(0)
                t0, got = time.time(), 0
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    got += len(block)
                    if got % (100 * CHUNK) < CHUNK:
                        rate = got / max(time.time() - t0, 1e-6) / 1e6
                        print(f"  {dest.name}: {(have + got) / 1e6:.0f}/{size / 1e6:.0f} MB ({rate:.1f} MB/s)",
                              flush=True)
        except Exception as e:  # network hiccup: back off and resume
            print(f"  {dest.name}: attempt {attempt + 1} failed ({e}); resuming", flush=True)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed to download {dest.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--participants", type=int, nargs="*", default=list(range(1, 13)))
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "zips").mkdir(parents=True, exist_ok=True)
    files = list_participant_files()
    manifest = {}
    for n in args.participants:
        rec = files[n]
        dest = out / "zips" / rec["name"]
        md5 = rec.get("computed_md5") or rec.get("supplied_md5")
        print(f"P{n}: {rec['name']} {rec['size'] / 1e6:.0f} MB", flush=True)
        download(rec["download_url"], dest, rec["size"])
        got = md5sum(dest)
        if got != md5:
            dest.unlink()
            raise RuntimeError(f"MD5 mismatch for {dest.name}: {got} != {md5}; deleted, rerun to retry")
        if not args.no_extract:
            target = out / f"P{n}"
            if not target.exists():
                with zipfile.ZipFile(dest) as z:
                    for m in z.namelist():  # guard against path traversal
                        assert ".." not in Path(m).parts and not Path(m).is_absolute(), m
                    z.extractall(target)
        manifest[n] = {"file": rec["name"], "size": rec["size"], "md5": md5, "url": rec["download_url"]}
        print(f"P{n}: ok", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"verified {len(manifest)} participants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
