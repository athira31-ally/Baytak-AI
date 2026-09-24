"""Peek at data.dubai's DLD files without downloading them: for every part, read the first 64 KB
(one HTTP Range request) and report compression, columns, a few rows and the date range at its start.
Tells us whether the parts are in date order (so a refresh can read only the newest parts).

    python -m scripts.peek_dld rents        # Ejari rent contracts (dataset 468586)
    python -m scripts.peek_dld sales        # real estate transactions (dataset 470061)
"""
import csv
import io
import sys
import zlib

import httpx
import pandas as pd

from app.data.agent import DATASET_ID, RENT_DATASET_ID, download_links
from app.data.dld import BROWSER_UA, pick, to_dt


def head_bytes(url: str, n: int = 65536):
    r = httpx.get(url, headers={"User-Agent": BROWSER_UA, "Range": f"bytes=0-{n - 1}",
                                "Accept-Encoding": "identity"}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.status_code, r.headers, r.content


def main() -> None:
    which = (sys.argv[1] if len(sys.argv) > 1 else "rents").lower()
    ds = RENT_DATASET_ID if which.startswith("rent") else DATASET_ID
    links = download_links(dataset_id=ds)
    print(f"dataset {ds}: {len(links)} CSV parts")
    for i, (name, url) in enumerate(sorted(links.items())):
        status, h, b = head_bytes(url)
        gz = b[:2] == b"\x1f\x8b"
        text = zlib.decompressobj(31).decompress(b) if gz else b
        lines = text.decode("utf-8", "replace").splitlines()[:-1]
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
        df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame()
        col = pick(df, "date") if len(df) else None
        d = to_dt(df[col]) if col else pd.Series(dtype="datetime64[ns]")
        print(f"\n[{i + 1}] {name}  http={status}  gzip_bytes={gz}  content-encoding={h.get('content-encoding')}  "
              f"total={h.get('content-range', '').split('/')[-1]} bytes")
        print(f"    first rows' {col}: {d.min().date() if len(d.dropna()) else '?'} .. {d.max().date() if len(d.dropna()) else '?'}"
              f"  ({len(df)} rows sampled)")
        if i == 0:
            print("    columns:", list(df.columns))
            print(df.head(3).T.to_string()[:2500])


if __name__ == "__main__":
    main()
