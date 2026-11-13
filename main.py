"""
Setup:   pip install requests python-dotenv
.env:    OPENROUTER_API_KEY=...   OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
Run:     python main.py [--input data/Input.json] [--output data/output/normalized_output.csv]
         python main.py --no-llm     (rules only, no API calls)

Pipeline: rule-based Python (switcher, finance, unique discount, service line)
          + LLM (fuzzy fields) -> validation -> exact 29-column CSV, 2 rows/product.
"""
import argparse, csv, json, os, re, sys, time
from decimal import Decimal
from pathlib import Path
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

COLUMNS = [
    "PLAN TYPE", "PRODUCT NAME", "BRAND", "VARIANT", "BASE", "MSRP",
    "RESELLER LIST PRICE", "SWITCHER", "SWITCHER_OFFER_TEXT",
    "UNIQUE DISCOUNT", "FREE MEMORY UPGRADE", "BOGO OFFER",
    "Max TI (Maximum Trade-In Value)", "Trade-In Value Type (Enhanced/ Base)",
    "LINE TYPE", "RATEPLAN DURATION (MONTHS)", "SHIPMENT", "STOCK",
    "PWP/GWP OFFERS (Pay-With-Purchase / Gift-With-Purchase)",
    "Cleaned PWP/GWP OFFERS", "PWP/GWP OFFER $ VALUE",
    "SUBSCRIPTION BUNDLE OFFERS", "SUBSCRIPTION BUNDLE OFFER $ VALUE",
    "SERVICE LINE OFFERS", "SERVICE LINE OFFER $ VALUE",
    "ADDITIONAL OFFERS", "ADDITIONAL OFFER $ VALUE",
    "FINANCE OFFER DESCRIPTION 1", "FINANCE OFFER $ VALUE",
]

# Fields the LLM is allowed to fill (everything else is computed in Python)
LLM_FIELDS = [
    "FREE MEMORY UPGRADE", "BOGO OFFER", "Max TI (Maximum Trade-In Value)",
    "Trade-In Value Type (Enhanced/ Base)",
    "PWP/GWP OFFERS (Pay-With-Purchase / Gift-With-Purchase)",
    "Cleaned PWP/GWP OFFERS", "PWP/GWP OFFER $ VALUE",
    "SUBSCRIPTION BUNDLE OFFERS", "SUBSCRIPTION BUNDLE OFFER $ VALUE",
    "ADDITIONAL OFFERS", "ADDITIONAL OFFER $ VALUE",
]

MONEY_FIELDS = {
    "SWITCHER", "UNIQUE DISCOUNT", "Max TI (Maximum Trade-In Value)",
    "PWP/GWP OFFER $ VALUE", "SUBSCRIPTION BUNDLE OFFER $ VALUE",
    "SERVICE LINE OFFER $ VALUE", "ADDITIONAL OFFER $ VALUE",
    "FINANCE OFFER $ VALUE", "BASE", "MSRP", "RESELLER LIST PRICE",
}

NEW_LINE, UPGRADE = "New Line", "Upgrade"


# ----------------------------------------------------------------- helpers
def get(obj, *paths):
    for path in paths:
        v = obj
        for part in path.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if v not in (None, ""):
            return v
    return ""


def format_money(value):
    """'$' + exact decimals as found in the source. No rounding, no thousands
    separators, no surrounding whitespace."""
    if value in (None, ""):
        return ""
    m = re.search(r"-?[\d,]+(?:\.\d+)?", str(value))
    if not m:
        return ""
    amount = Decimal(m.group(0).replace(",", ""))
    decimals = max(0, -amount.as_tuple().exponent)
    return f"${amount:.{decimals}f}"


def money_sum(*vals):
    return sum((Decimal(v) for v in vals), Decimal(0))


def offer_texts(product):
    """Each offer's title/description/modal text, de-duplicated, no accessories."""
    out = []
    for o in product.get("offers") or []:
        fd = o.get("full_details") or {}
        parts = [o.get("title"), o.get("description"), fd.get("modal_raw_text")]
        for c in fd.get("cards") or []:
            parts += [c.get("title"), c.get("legal")]
            parts.append((c.get("full_details") or {}).get("raw_text"))
        txt = "\n".join(p for p in parts if p)
        if txt and txt not in out:
            out.append(txt)
    return out


# ------------------------------------------------------- rule-based fields
def variant_pattern(variant):
    m = re.match(r"\s*(\d+)\s*(GB|TB)", str(variant), re.I)
    return rf"(?<!\d){m.group(1)}\s*{m.group(2)}" if m else re.escape(str(variant))


def rule_fields(product, line_type, all_text):
    r = {}
    is_new = line_type == NEW_LINE

    # SWITCHER (port-in => New Line only)
    m = re.search(r"up to (?:an )?\$(\d[\d,]*)(?:/line| (?:AT&T )?Visa Reward)", all_text, re.I)
    if m and is_new:
        n = m.group(1).replace(",", "")
        r["SWITCHER"] = n
        r["SWITCHER_OFFER_TEXT"] = (f"Get up to ${n}/line when you switch to AT&T "
                                    "and get a reward card to pay off your phone.")

    # Global "new line only" gate for the device-credit block
    new_only = re.search(r"Eligibility:\s*New customers activating new line", all_text, re.I)

    # UNIQUE DISCOUNT: variant-anchored bill credits
    if is_new or not new_only:
        pat = (variant_pattern(product.get("variant", "")) +
               r"\s+for\s+\$[\d.,]+/mo\.\s+for\s+\d+\s+mos\.\s+after\s+up\s+to\s+"
               r"\$([\d,]+(?:\.\d+)?)\s+in\s+bill\s+credits")
        m = re.search(pat, all_text, re.I)
        if m:
            r["UNIQUE DISCOUNT"] = m.group(1).replace(",", "")

    # SERVICE LINE credit (new consumer lines only)
    if is_new:
        m = re.search(r"additional \$(\d+) in credits", all_text, re.I)
        if m:
            r["SERVICE LINE OFFERS"] = f"Price after additional ${m.group(1)} in credits."
            r["SERVICE LINE OFFER $ VALUE"] = m.group(1)
        else:
            m = re.search(r"Get \$(\d+) off a new line", all_text, re.I)
            if m:
                r["SERVICE LINE OFFERS"] = f"Get ${m.group(1)} off a new line."
                r["SERVICE LINE OFFER $ VALUE"] = m.group(1)

    # FINANCE (Citi card): welcome bonus only for New Line; recurring for both
    if "Points Plus" in all_text:
        w = re.search(r"Earn a \$(\d+) (?:welcome bonus|statement credit)", all_text)
        y = re.search(r"up to \$(\d+) in statement credits every year", all_text)
        d = re.search(r"\$(\d+) per line monthly discount", all_text) or \
            re.search(r"Earn \$(\d+) discounts on your AT&T bills", all_text)
        yearly = y.group(1) if y else None
        disc = d.group(1) if d else None
        welcome = w.group(1) if (w and is_new) else None
        vals = [v for v in (welcome, yearly, disc) if v]
        if vals:
            r["FINANCE OFFER $ VALUE"] = str(money_sum(*vals))
            tail = []
            if yearly:
                tail.append(f"up to ${yearly} back every year with the AT&T Points Plus® Card from Citi")
            if disc:
                tail.append(f"${disc} discounts on your AT&T bills")
            tail = " and ".join(tail)
            r["FINANCE OFFER DESCRIPTION 1"] = (
                f"Earn a ${welcome} statement credit after qualifying purchases welcome bonus plus {tail}"
                if welcome else f"Get {tail}")
    return r


# --------------------------------------------------------------- LLM phase
def llm_payload(product):
    skip = ("statement credit", "break your contract")
    texts = [t for t in offer_texts(product) if not any(s in t[:200].lower() for s in skip)]
    return {
        "product_name": product.get("product_name"),
        "variant": product.get("variant"),
        "pricing": product.get("pricing"),
        "offer_texts": [t[:6000] for t in texts],
    }


def parse_json(content):
    """Tolerant JSON parser: handles None, code fences, text around the object."""
    if not content or not isinstance(content, str):
        return None
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def llm_extract(product, line_type, model, retries=3):
    empty = {f: "" for f in LLM_FIELDS}
    instructions = f"""Extract promotions from ONLY the supplied data. Return a single JSON object
with exactly these keys: {json.dumps(LLM_FIELDS)}. Use "" for missing/ineligible values.
Rules:
- Evaluate as LINE TYPE = {line_type}. Offers requiring a new line/port-in/new customer are NOT eligible for Upgrade.
- Only use offers for this exact variant ({product.get('variant')}); never another variant's.
- Max TI requires an explicit maximum dollar trade-in amount (e.g. "up to $930 off"); generic trade-in
  instructions without a dollar maximum -> "". Trade-In Value Type: Enhanced/Base only if stated.
- Never put one offer/amount in more than one category. Do not sum values.
- Do not handle switcher, unique discount, service-line, bill-credit or Citi card offers (handled elsewhere).
- ADDITIONAL OFFERS: only promotions not covered by any other category. Paid add-ons such as
  Next Up Anytime are not offers. Never compute, multiply or sum amounts; only copy amounts stated in the text.
- Never guess or use outside knowledge. Keep exact decimals. Output JSON only."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                    "X-Title": "Product Offer Extractor",
                },
                json={"model": model, "temperature": 0, "max_tokens": 1500,
                      "messages": [
                          {"role": "system", "content": instructions},
                          {"role": "user", "content": json.dumps(llm_payload(product), ensure_ascii=False)}]},
                timeout=120,
            )
            if resp.status_code == 404:           # model slug problem: retrying won't help
                sys.exit(f"Model unavailable: {resp.text}")
            resp.raise_for_status()
            body = resp.json()
            if "error" in body:
                raise RuntimeError(body["error"])
            data = parse_json(body["choices"][0]["message"].get("content"))
            if isinstance(data, dict):
                return {f: ("" if data.get(f) is None else str(data.get(f))) for f in LLM_FIELDS}
            raise ValueError("empty/malformed LLM output")
        except (requests.RequestException, RuntimeError, ValueError, KeyError, IndexError) as e:
            print(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    print("  -> giving up on LLM fields for this row (left blank)")
    return empty


# -------------------------------------------------------------- validation
def _num(v):
    """First number in a string as Decimal, or None."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(v or ""))
    return Decimal(m.group(0).replace(",", "")) if m else None


def _plain(n):
    """Decimal -> plain string without trailing zeros ('930', '5.56')."""
    s = format(n, "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def validate(row, all_text):
    """Post-LLM guardrails: remove invented, zero or duplicated values."""
    src = all_text.replace(",", "")

    # 1. LLM money must literally appear in the source text and be non-zero
    for f in ("Max TI (Maximum Trade-In Value)", "PWP/GWP OFFER $ VALUE",
              "SUBSCRIPTION BUNDLE OFFER $ VALUE", "ADDITIONAL OFFER $ VALUE"):
        n = _num(row.get(f))
        if n is None or n == 0 or _plain(n) not in src:
            row[f] = ""

    # 2. Additional offer must not duplicate another category or be a fee/headline
    add = _num(row.get("ADDITIONAL OFFER $ VALUE"))
    others = [_num(row.get(k)) for k in (
        "SERVICE LINE OFFER $ VALUE", "UNIQUE DISCOUNT", "SWITCHER", "FINANCE OFFER $ VALUE")]
    txt = str(row.get("ADDITIONAL OFFERS", "")).lower()
    if (add is not None and add in others) or any(
            k in txt for k in ("next up anytime", "bill credit", "off a new line", "for $0")):
        row["ADDITIONAL OFFERS"] = row["ADDITIONAL OFFER $ VALUE"] = ""

    # 3. Rate-plan duration: deterministic, same rule for both line types
    m = (re.search(r"(\d+)-month installment", all_text, re.I)
         or re.search(r"for (\d+) mos", all_text, re.I)
         or re.search(r"after (\d+) monthly bill credits", all_text, re.I))
    row["RATEPLAN DURATION (MONTHS)"] = m.group(1) if m else ""
    return row


# ------------------------------------------------------------- row builder
def make_row(product, line_type, model, use_llm=True):
    all_text = "\n".join(offer_texts(product))
    pricing = product.get("pricing") or {}
    row = {c: "" for c in COLUMNS}
    row.update({
        "PLAN TYPE": get(product, "plan_type"),
        "PRODUCT NAME": get(product, "product_name", "name"),
        "BRAND": get(product, "brand"),
        "VARIANT": get(product, "variant"),
        "BASE": get(pricing, "monthly"),
        "MSRP": get(pricing, "retail"),
        "RESELLER LIST PRICE": get(product, "reseller_list_price"),
        "SHIPMENT": get(product, "shipment"),
        "STOCK": get(product, "stock"),
    })
    if use_llm:
        row.update(llm_extract(product, line_type, model))
    row.update(rule_fields(product, line_type, all_text))   # rules override the LLM
    row["LINE TYPE"] = line_type
    validate(row, all_text)

    # Upgrade can never carry new-line-only offers
    if line_type == UPGRADE:
        for f in ("SWITCHER", "SWITCHER_OFFER_TEXT"):
            row[f] = ""
    for f in MONEY_FIELDS:
        row[f] = format_money(row.get(f))
    return {c: str(row.get(c, "")).strip() for c in COLUMNS}


def read_products(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("products") or data.get("items") or data.get("data")
    if not isinstance(data, list):
        raise ValueError("JSON must be a product array or contain products/items/data.")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Input.json")
    ap.add_argument("--output", default="data/output/normalized_output.csv")
    ap.add_argument("--no-llm", action="store_true", help="rules only (no API calls)")
    args = ap.parse_args()

    use_llm = not args.no_llm
    if use_llm and not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY in .env (or run with --no-llm).")
    if not Path(args.input).exists():
        sys.exit(f"Input file not found: {args.input}")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

    products = [p for p in read_products(args.input) if isinstance(p, dict)]
    rows = []
    for i, p in enumerate(products, 1):
        print(f"Processing {i}/{len(products)}: {get(p, 'product_name')} {get(p, 'variant')}")
        for lt in (NEW_LINE, UPGRADE):
            rows.append(make_row(p, lt, model, use_llm))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:   # no BOM: headers stay exact
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert len(rows) == len(products) * 2, "Row count validation failed"
    print(f"Done: {len(rows)} rows, {len(COLUMNS)} columns -> {out}")


if __name__ == "__main__":
    main()