"""Check your Dubai Pulse API access before turning on live data.

    python -m scripts.check_dubai_pulse          # reads DUBAI_PULSE_API_KEY / _SECRET from .env

Prints: whether the token works, the columns the API returns, whether date filtering
works, and how many homes one page of data turns into.
"""
import sys
from datetime import date, timedelta

import httpx

from app.config import get_settings
from app.data.dld import ColumnError, build
from app.data.live import DubaiPulseClient


def main() -> None:
    s = get_settings()
    if not s.use_live_data:
        sys.exit("Set DUBAI_PULSE_API_KEY and DUBAI_PULSE_API_SECRET in .env first.")
    c = DubaiPulseClient(s.dubai_pulse_api_key, s.dubai_pulse_api_secret, page_size=1000)
    try:
        c.token()
        print("1. Token: OK")
    except httpx.HTTPError as e:
        sys.exit(f"1. Token FAILED: {e}\n   Check the key/secret emails from Dubai Pulse.")

    for name, url, col in [("sales", s.dubai_pulse_sales_url, "instance_date"),
                           ("rents", s.dubai_pulse_rents_url, "contract_start_date")]:
        if not url:
            continue
        try:
            r = c.http.get(url, params={"limit": 5, "offset": 0}, headers={"Authorization": f"Bearer {c.token()}"})
            r.raise_for_status()
            rows = c._rows(r)
            print(f"2. {name}: OK, sample columns: {sorted(rows[0])[:25] if rows else 'no rows'}")
        except Exception as e:
            print(f"2. {name}: FAILED {url}\n   {e}\n   Copy the API URL from the dataset page on dubaipulse.gov.ae into "
                  f"DUBAI_PULSE_{name.upper()}_URL in .env")
            continue
        try:
            df = c.fetch(url, col, date.today() - timedelta(days=30), max_rows=2000)
            print(f"3. {name}: fetched {len(df):,} rows from the last 30 days")
            if name == "sales":
                homes, _ = build(df, None, months=1, min_deals=1, verbose=True)
                print(f"4. Built {len(homes):,} homes from that sample. Live data is ready to switch on.")
        except ColumnError as e:
            print(f"4. Column mismatch - paste this to Claude:\n{e}")
        except Exception as e:
            print(f"3. {name}: paging failed: {e}")


if __name__ == "__main__":
    main()
