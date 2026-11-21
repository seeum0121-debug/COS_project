"""Download only the reviewed public URLs in V5/수집결과.json.

Default invocation lists files. --download performs network requests.
No authentication, browser session, proxy alteration or access-control bypass is used.
Existing files are never overwritten. ZIP archives are not extracted.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
ROOT = BASE / "원본파일"
MANIFEST = BASE / "수집결과.json"
LOG = BASE / "다운로드_실행기록.jsonl"

def validate(path, kind):
    with path.open("rb") as f:
        head = f.read(4096)
    if not head:
        raise ValueError("Empty response")
    lower = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if lower.startswith((b"<!doctype html", b"<html")) or b"<html" in lower[:200]:
        raise ValueError("HTML response is not a data file")
    kind = kind.upper()
    if kind in ("ZIP", "XLSX"):
        if not zipfile.is_zipfile(path):
            raise ValueError("Invalid ZIP/XLSX signature")
        with zipfile.ZipFile(path) as z:
            if not z.infolist():
                raise ValueError("Empty archive")
            if kind == "XLSX" and "[Content_Types].xml" not in z.namelist():
                raise ValueError("ZIP is not an XLSX workbook")
        # This validates the container, not every entry's CRC or semantic contents.
    elif kind == "PDF" and not head.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF signature")
    elif kind == "CSV":
        if b"\x00" in head:
            raise ValueError("Binary/NUL response is not expected CSV")
        if lower.startswith((b"{", b"[")):
            raise ValueError("JSON response returned for CSV")
        if b"," not in head and b";" not in head and b"\t" not in head:
            raise ValueError("No recognizable CSV delimiter in header")
    elif kind == "JSON":
        json.loads(path.read_text(encoding="utf-8-sig"))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_one(row, max_bytes):
    filename = row["filename"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Manifest filename must be a plain basename")
    ROOT.mkdir(exist_ok=True)
    target = (ROOT / filename).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("Output outside V5/원본파일")
    if target.exists():
        validate(target, row["format"])
        return {
            "status": "existing_valid_container",
            "path": str(target.relative_to(BASE)), "bytes": target.stat().st_size,
            "sha256": digest(target), "note": "Existing file; not re-fetched or compared to current server",
        }
    parsed = urllib.parse.urlsplit(row["url"])
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only reviewed HTTPS URLs are allowed")
    temporary = ROOT / (filename + "." + uuid.uuid4().hex + ".part")
    try:
        req = urllib.request.Request(row["url"], headers={"User-Agent": "V5-public-data-collection/1.0"})
        with urllib.request.urlopen(req, timeout=45) as response:
            if response.status != 200:
                raise ValueError("Unexpected HTTP status " + str(response.status))
            final_url = response.geturl()
            if urllib.parse.urlsplit(final_url).scheme != "https":
                raise ValueError("Refusing downgrade redirect")
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise ValueError("Server returned HTML, possibly access denied/challenge")
            declared = response.headers.get("Content-Length")
            expected_length = int(declared) if declared and declared.isdigit() else None
            if expected_length is not None and expected_length > max_bytes:
                raise ValueError("Content-Length exceeds --max-mb limit")
            size = 0
            h = hashlib.sha256()
            with temporary.open("xb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError("Download exceeds --max-mb limit")
                    output.write(block)
                    h.update(block)
            if expected_length is not None and size != expected_length:
                raise ValueError("Incomplete response: Content-Length mismatch")
        validate(temporary, row["format"])
        # Exclusive creation protects existing files even if another process races us.
        try:
            os.link(temporary, target)
            temporary.unlink()
        except OSError:
            if target.exists():
                raise FileExistsError("Target appeared during download; not overwritten")
            with target.open("xb") as output, temporary.open("rb") as input_file:
                for block in iter(lambda: input_file.read(1024 * 1024), b""):
                    output.write(block)
            temporary.unlink()
        return {
            "status": "downloaded_container_validated",
            "path": str(target.relative_to(BASE)), "bytes": size,
            "sha256": h.hexdigest(), "final_url": final_url, "content_type": content_type,
            "note": "Container signature checked; dataset semantics and archive entry CRC not validated",
        }
    finally:
        # This UUID .part path was created by this invocation only; no recursive deletion.
        if temporary.exists():
            temporary.unlink()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="List reviewed URLs without network access")
    ap.add_argument("--download", action="store_true", help="Download selected files")
    ap.add_argument("--id", action="append", default=[], help="Download ID, may be repeated; otherwise default bundle")
    ap.add_argument("--max-mb", type=int, default=2048, help="Maximum bytes per file in MiB (default 2048)")
    args = ap.parse_args()
    if args.max_mb < 1:
        ap.error("--max-mb must be positive")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = data["downloads"]
    by_id = {r["download_id"]: r for r in rows}
    unknown = set(args.id) - set(by_id)
    if unknown:
        ap.error("Unknown IDs: " + ", ".join(sorted(unknown)))
    selected = [r for r in rows if r["download_id"] in args.id] if args.id else [r for r in rows if r["default_download"]]
    if args.list or not args.download:
        display = selected if args.id else rows
        for r in display:
            default = "*" if r["default_download"] else " "
            print(f'{default} {r["download_id"]:10} {r["format"]:5} {r["filename"]}')
            print("  " + r["url"])
        print(f"{len(rows)} reviewed URLs; {len(selected)} selected. * = default bundle.")
        print("No network requests performed. Original download status in workbook remains unchanged.")
        return 0
    failed = 0
    for row in selected:
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "download_id": row["download_id"], "source_id": row["source_id"], "url": row["url"],
        }
        try:
            record.update(download_one(row, args.max_mb * 1024 * 1024))
        except Exception as exc:
            failed += 1
            record.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(row["download_id"] + ": " + record["status"], flush=True)
        if record["status"] == "failed":
            print("  " + record["error"], flush=True)
        # Modest pacing; no retry loops or access-control circumvention.
        time.sleep(0.25)
    print(f"Finished: {len(selected) - failed} valid existing/downloaded, {failed} failed. Log: {LOG.name}")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())

