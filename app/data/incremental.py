"""Fetch ONLY the deals registered since the last run.

DLD publishes the full history as two ~550 MB CSVs that are regenerated daily. Re-reading
1.1 GB to get one day of new deals is wasteful, so we use HTTP Range requests:

  1. read a few KB at the start and end of each file to learn the header and whether
     rows are sorted oldest->newest or newest->oldest;
  2. read 4 MB blocks from the NEWEST end until a block reaches deals older than the
     watermark, then stop.

A daily top-up is then a few MB instead of 1.1 GB. If the server doesn't support Range
requests (or rows aren't date-sorted), it falls back to one streamed pass, still keeping
only new rows. The Dubai Pulse API path (filter by date) is used instead when keys are set.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass

import httpx
import pandas as pd

from app.data.dld import BROWSER_UA, WANTED, _chunks, _snake, pick, to_dt

log = logging.getLogger(__name__)
BLOCK = 4 << 20           # 4 MB per range request (a daily top-up is usually 1-3 requests)
PROBE = 256 << 10         # 256 KB to sniff header / first and last dates


@dataclass
class FetchStats:
    bytes_read: int = 0
    requests: int = 0
    mode: str = ""


class RangeFile:
    def __init__(self, url: str, http: httpx.Client, stats: FetchStats):
        self.url, self.http, self.stats = url, http, stats
        head = http.head(url, follow_redirects=True)
        head.raise_for_status()
        self.size = int(head.headers.get("content-length", 0))
        self.ranges = head.headers.get("accept-ranges", "").lower() == "bytes" and self.size > 0

    def read(self, start: int, end: int) -> bytes:           # [start, end)
        r = self.http.get(self.url, headers={"Range": f"bytes={start}-{end - 1}"}, follow_redirects=True)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("server ignored the Range header")
        self.stats.bytes_read += len(r.content)
        self.stats.requests += 1
        return r.content


def _parse(text: str, header: list[str]) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), header=None, names=header, low_memory=False, on_bad_lines="skip")
    return df[[c for c in df.columns if _snake(c) in WANTED]]


def _tail_since(f: RangeFile, header: list[str], header_len: int, date_col: str, since: pd.Timestamp) -> pd.DataFrame:
    """Rows sorted oldest->newest: walk backwards from the end."""
    parts, end = [], f.size
    while end > header_len:
        start = max(header_len, end - BLOCK)
        raw = f.read(start, end)
        cut = 0
        if start > header_len:                                  # drop the partial first line
            cut = raw.index(b"\n") + 1
        df = _parse(raw[cut:].decode("utf-8", "replace"), header)
        d = to_dt(df[date_col])
        parts.append(df[d >= since])
        if d.min() < since or start == header_len:
            break
        end = start + cut
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=header)


def _head_since(f: RangeFile, header: list[str], header_len: int, date_col: str, since: pd.Timestamp) -> pd.DataFrame:
    """Rows sorted newest->oldest: walk forwards from the start."""
    parts, start = [], header_len
    while start < f.size:
        end = min(f.size, start + BLOCK)
        raw = f.read(start, end)
        last_nl = raw.rfind(b"\n") + 1 if end < f.size else len(raw)
        df = _parse(raw[:last_nl].decode("utf-8", "replace"), header)
        d = to_dt(df[date_col])
        parts.append(df[d >= since])
        if d.max() < since or end == f.size:
            break
        start += last_nl
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=header)


def fetch_since_files(urls: list[str], since: pd.Timestamp) -> tuple[pd.DataFrame, FetchStats]:
    stats = FetchStats()
    out = []
    with httpx.Client(timeout=httpx.Timeout(60, read=300), headers={"User-Agent": BROWSER_UA}) as http:
        for url in urls:
            if not url.startswith("http"):                      # local file: stream it once
                out.append(_stream_since(url, since))
                stats.mode = "full scan (local file)"
                continue
            f = RangeFile(url, http, stats)
            if not f.ranges:
                log.info("%s: no Range support, streaming once", url)
                out.append(_stream_since(url, since))
                stats.mode, stats.bytes_read = "full scan (no Range support)", stats.bytes_read + f.size
                continue
            first = f.read(0, min(PROBE, f.size))
            header_line = first.split(b"\n", 1)[0]
            header = next(csv.reader([header_line.decode("utf-8-sig")]))
            date_col = pick(pd.DataFrame(columns=header), "date")
            header_len = len(header_line) + 1
            first_rows = _parse(first[header_len:first.rfind(b"\n") + 1].decode("utf-8", "replace"), header)
            last = f.read(max(header_len, f.size - PROBE), f.size)
            last_rows = _parse(last[last.index(b"\n") + 1:].decode("utf-8", "replace"), header)
            d_first, d_last = to_dt(first_rows[date_col]).median(), to_dt(last_rows[date_col]).median()
            if not (_sorted(first_rows[date_col]) and _sorted(last_rows[date_col])):
                log.info("%s: rows not date-sorted, streaming once", url)   # can't safely stop early
                out.append(_stream_since(url, since))
                stats.mode, stats.bytes_read = "full scan (rows not date-sorted)", stats.bytes_read + f.size
                continue
            if d_last >= d_first:
                if to_dt(last_rows[date_col]).max() < since:
                    continue                                    # this part has nothing new
                out.append(_tail_since(f, header, header_len, date_col, since))
                stats.mode = "range: tail (file sorted oldest->newest)"
            else:
                if to_dt(first_rows[date_col]).max() < since:
                    continue
                out.append(_head_since(f, header, header_len, date_col, since))
                stats.mode = "range: head (file sorted newest->oldest)"
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return df, stats


def _sorted(dates: pd.Series, tolerance: float = 0.05) -> bool:
    """True if the rows are (nearly) monotonic by date, ascending or descending."""
    d = to_dt(dates).dropna()
    if len(d) < 10:
        return True
    diff = d.diff().dropna()
    return min((diff < pd.Timedelta(0)).mean(), (diff > pd.Timedelta(0)).mean()) <= tolerance


def _stream_since(src: str, since: pd.Timestamp) -> pd.DataFrame:
    keep, date_col = [], None
    for chunk in _chunks(src, 250_000):
        date_col = date_col or pick(chunk, "date")
        keep.append(chunk[to_dt(chunk[date_col]) >= since])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
