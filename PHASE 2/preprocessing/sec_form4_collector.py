"""Conservative SEC Form 4 downloader for one US ticker."""
import argparse, json, os, time, re
from pathlib import Path
from urllib.parse import quote
import requests
import pandas as pd
from dotenv import load_dotenv
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/sec"
OUT = ROOT / "data/processed/sec_form4_transactions.csv"
S = requests.Session()
S.headers.update({"Accept-Encoding": "gzip, deflate"})
last = 0.0

def validate_user_agent():
    load_dotenv(ROOT / ".env")
    ua = os.getenv("SEC_USER_AGENT", "")
    if not ua or "your-email@example.com" in ua:
        raise SystemExit("Set SEC_USER_AGENT in .env to an app name and real contact email.")
    return ua

def get(url):
    global last
    if "User-Agent" not in S.headers:
        ua = validate_user_agent()
        S.headers.update({"User-Agent": ua})
    delay = 0.20 - (time.monotonic() - last)
    if delay > 0: time.sleep(delay)
    r = S.get(url, timeout=40)
    last = time.monotonic()
    r.raise_for_status()
    return r

def local_text(node, path):
    found = node.xpath(path)
    return "".join(found[0].itertext()).strip() if found else ""

def main(ticker, limit):
    ua = validate_user_agent()
    S.headers.update({"User-Agent": ua})
    RAW.mkdir(parents=True, exist_ok=True)
    mapping_path = RAW / "company_tickers.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
    else:
        r = get("https://www.sec.gov/files/company_tickers.json")
        mapping_path.write_bytes(r.content)
        mapping = r.json()
    item = next((x for x in mapping.values() if x["ticker"].upper() == ticker.upper()), None)
    if not item: raise ValueError(f"Ticker not found: {ticker}")
    cik = str(item["cik_str"]).zfill(10)
    submissions = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    recent = submissions.get("filings", {}).get("recent", {})
    filings = []
    for i, form in enumerate(recent.get("form", [])):
        if form == "4":
            filings.append({k: recent.get(k, [""] * len(recent["form"]))[i]
                            for k in ["filingDate", "reportDate", "accessionNumber", "primaryDocument"]})
    filings = filings[:limit]
    target_dir = RAW / ticker.upper()
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for filing in filings:
        accession = filing["accessionNumber"]
        doc = filing["primaryDocument"]
        if not accession or not doc: continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{quote(doc)}"
        try:
            response = get(url)
        except requests.RequestException as e:
            print("Download failed:", accession, e)
            continue
        raw_path = target_dir / f"{accession}_{Path(doc).name}"
        raw_path.write_bytes(response.content)
        parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
        try: root = etree.fromstring(response.content, parser)
        except etree.XMLSyntaxError:
            print("Not parseable XML, preserved raw:", raw_path.name)
            continue
        if "ownershipDocument" not in etree.QName(root).localname:
            print("Not a direct ownership XML, preserved raw:", raw_path.name)
            continue
        issuer_name = local_text(root, "//*[local-name()='issuerName']")
        symbol = local_text(root, "//*[local-name()='issuerTradingSymbol']") or ticker.upper()
        owners = root.xpath("//*[local-name()='reportingOwner']")
        names, roles = [], []
        for owner in owners:
            names.append(local_text(owner, ".//*[local-name()='rptOwnerName']"))
            rel = owner.xpath(".//*[local-name()='reportingOwnerRelationship']")
            if rel:
                title = local_text(rel[0], ".//*[local-name()='officerTitle']")
                flags = [tag for tag in ["isDirector","isOfficer","isTenPercentOwner"]
                         if local_text(rel[0], f"./*[local-name()='{tag}']").lower() in ("1","true")]
                roles.extend(flags + ([title] if title else []))
        for table_name, tx_name in [("nonDerivativeTable","nonDerivativeTransaction"),
                                    ("derivativeTable","derivativeTransaction")]:
            for tx in root.xpath(f"//*[local-name()='{table_name}']/*[local-name()='{tx_name}']"):
                def t(path): return local_text(tx, path)
                code = t(".//*[local-name()='transactionCoding']/*[local-name()='transactionCode']")
                shares = t(".//*[local-name()='transactionShares']/*[local-name()='value']")
                price = t(".//*[local-name()='transactionPricePerShare']/*[local-name()='value']")
                acquired_disposed = t(".//*[local-name()='transactionAcquiredDisposedCode']/*[local-name()='value']")
                after = t(".//*[local-name()='sharesOwnedFollowingTransaction']/*[local-name()='value']")
                date = t(".//*[local-name()='transactionDate']/*[local-name()='value']")
                try: sh = float(shares) if shares else None
                except ValueError: sh = None
                try: px = float(price) if price else None
                except ValueError: px = None
                rows.append({
                    "ticker": symbol, "issuer_name": issuer_name, "cik": cik,
                    "filing_date": filing["filingDate"], "report_date": filing["reportDate"],
                    "accession_number": accession, "filing_document": doc,
                    "insider_names": "; ".join(x for x in names if x),
                    "insider_roles": "; ".join(sorted(set(roles))),
                    "security_table": table_name, "transaction_date": date,
                    "transaction_code": code, "acquired_disposed": acquired_disposed,
                    "shares": sh, "price_per_share": px,
                    "estimated_value": sh * px if sh is not None and px is not None else None,
                    "shares_owned_after": after, "filing_url": url,
                    "mentions_10b5_1": ("10b5-1" in response.text.lower() or "10b5 1" in response.text.lower()),
                    "is_sale_code_S": code == "S"
                })
    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows)
        if OUT.exists():
            new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
            new = new.drop_duplicates(["accession_number","transaction_date","transaction_code","shares","price_per_share"])
        new.to_csv(OUT, index=False)
    print(f"Saved {len(rows)} transaction row(s). Output: {OUT}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args()
    if not 1 <= a.limit <= 100: p.error("--limit must be 1..100")
    main(a.ticker, a.limit)
