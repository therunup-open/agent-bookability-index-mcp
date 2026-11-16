"""Read-only MCP server for the Agent Bookability Index (index.therunup.ai).

It reads only what the public site already publishes (data/vendors.json, prices/index.json and each vendor's
index.json), so it never shows more than the site does and needs no database or cloud account. Every answer
carries the index.therunup.ai page to cite and the date the vendor's site was read.

Tools (all read-only, all with typed output):
  search_vendors, get_vendor, shortlist_vendors, compare_vendors, get_price_summary, list_categories, and the
  search / fetch pair that ChatGPT connectors and deep research expect.

Run locally (stdio, for Claude Desktop or Claude Code):
    uv run python -m mcp_server.server
Run as a web service (streamable HTTP at /mcp, e.g. on a container host that sets PORT):
    uv run python -m mcp_server.server --http
INDEX_BASE_URL points it at another copy of the site (default https://index.therunup.ai).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, Any

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import Icon, ToolAnnotations
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, PlainTextResponse

BASE_URL = os.environ.get("INDEX_BASE_URL", "https://index.therunup.ai").rstrip("/")
TTL = 900  # the site can be republished several times a day; 15 minutes keeps answers fresh at little cost
MAX_LIMIT = 25
VERSION = "0.3.11"
CITE = "Agent Bookability Index by The Runup (index.therunup.ai)"
# Proof for the official MCP Registry that we control index.therunup.ai (namespace ai.therunup.index). It is the
# PUBLIC half of an Ed25519 key; the private half stays off the repo. Served at /.well-known/mcp-registry-auth.
REGISTRY_AUTH = "v=MCPv1; k=ed25519; p=BR9gMD1kczSjFiEJZ/3AVzUi7tU5HGcW21vJsw2RQAc="

AREAS = {  # area keys used on the site, with the words people use for them
    "manhattan": ["manhattan"],
    "brooklyn": ["brooklyn"],
    "queens": ["queens"],
    "bronx": ["bronx", "the bronx"],
    "staten-island": ["staten island"],
    "long-island": ["long island", "nassau", "suffolk", "the hamptons", "hamptons", "garden city", "huntington",
                    "great neck", "long beach"],
    "westchester": ["westchester", "white plains", "yonkers", "new rochelle", "rye", "scarsdale", "tarrytown"],
    "north-jersey": ["north jersey", "northern new jersey", "bergen", "hoboken", "jersey city", "essex county", "hudson county",
                     "passaic", "union county", "morris county", "sussex county", "warren county", "newark", "montclair", "morristown",
                     "paramus", "hackensack", "teaneck", "englewood", "fort lee", "paterson", "west orange",
                     "livingston", "short hills"],
    "central-jersey": ["central jersey", "central new jersey", "middlesex", "monmouth", "somerset county", "mercer county",
                       "ocean county", "hunterdon", "new brunswick", "princeton", "red bank", "toms river", "edison", "freehold",
                       "asbury park", "trenton", "long branch"],
}
CATEGORY_WORDS = {  # singular and loose words to the site's category keys
    "photographer": "photographers", "photography": "photographers", "photo": "photographers",
    "caterer": "caterers", "catering": "caterers", "food": "caterers",
    "florist": "florists", "flowers": "florists", "floral": "florists",
    "planner": "planners", "planning": "planners", "coordinator": "planners", "coordinators": "planners",
    "dj": "djs", "disc jockey": "djs",
    "band": "bands", "live band": "bands", "live bands": "bands", "musicians": "bands",
    "photo booth": "photo-booths", "photo booths": "photo-booths", "photobooth": "photo-booths",
    "videographer": "videographers", "video": "videographers", "videography": "videographers",
    "hair and makeup": "hair-and-makeup", "makeup": "hair-and-makeup", "hair": "hair-and-makeup",
    "cake": "cakes-and-desserts", "cakes": "cakes-and-desserts", "desserts": "cakes-and-desserts",
}
STOP_WORDS = {"best", "good", "great", "top", "cheap", "cheapest", "affordable", "find", "need", "looking", "want", "with",
              "that", "this", "what", "which", "where", "does", "much", "cost", "costs", "price", "prices", "pricing",
              "budget", "under", "below", "less", "than", "around", "about", "near", "nearby", "area", "york", "event",
              "events", "service", "services", "company", "companies", "book", "booking", "hire", "rent", "rental",
              "typical", "average", "median", "from", "their", "they", "your", "have", "some", "list", "recommend"}
EVENT_WORDS = {  # event types on the site, with the words people use for them
    "weddings": ["wedding", "weddings", "elopement", "engagement party", "rehearsal dinner"],
    "bar-bat-mitzvahs": ["bar mitzvah", "bat mitzvah", "b'nai mitzvah", "mitzvah"],
    "birthdays": ["birthday", "sweet 16", "sweet sixteen", "quinceanera", "quince", "anniversary"],
    "corporate": ["corporate", "company", "office", "holiday party", "work party", "conference", "gala"],
    "showers": ["shower", "baby shower", "bridal shower"],
    "other-private": ["party", "private event", "celebration", "reunion", "graduation"],
}

# ---------- typed results (each tool declares its output schema, so clients can validate and cite) ----------


class Price(BaseModel):
    amount_usd: float = Field(description="The vendor's lowest published starting price, US dollars.")
    per: str = Field("event", description='What the price is per: "event", "hour" or "guest".')
    covers: str | None = Field(None, description="What the price covers, in the index's words.")
    source: str | None = Field(None, description="The vendor's own page the price was read from.")


class Booking(BaseModel):
    online_booking: bool | None = Field(None, description="The vendor's site lets you book online.")
    shows_open_dates: bool | None = Field(None, description="The site shows dates that are still open.")
    instant_quote: bool | None = Field(None, description="The site shows a price after choosing options.")
    quote_request_form: bool | None = Field(None, description="The site has a form to ask for a quote.")
    contact_methods: list[str] = Field(default_factory=list,
                                       description="Ways to reach the vendor shown on its site (form, email, phone...).")


class Estimate(BaseModel):
    total_usd: float = Field(description="Starting price for this event, US dollars, from the vendor's published rate.")
    basis: str = Field(description='How it was worked out, e.g. "$35 per guest x 120 guests".')


class VendorCard(BaseModel):
    name: str
    category: str
    areas: list[str] = Field(description="Area keys the vendor serves.")
    score: int = Field(description="Bookability score 0-100: how much an AI agent can learn and do on the vendor's site.")
    starting_price: Price | None = None
    how_to_book: Booking
    website: str | None = None
    index_page: str = Field(description="The vendor's page on the index: cite this.")
    checked: str | None = Field(None, description="Date the vendor's website was read (YYYY-MM-DD).")
    estimate: Estimate | None = Field(None, description="Only in shortlists: the published starting price worked out "
                                                          "for this event's guests or hours. A starting point, not a quote.")


class SearchVendorsResult(BaseModel):
    category: str | None
    area: str | None
    matched: int = Field(description="How many vendors matched before the limit.")
    shown: int
    offset: int = Field(0, description="How many matches were skipped before these.")
    next_offset: int | None = Field(None, description="Pass this as offset to get the next page; None when there is "
                                                      "no more.")
    vendors: list[VendorCard]
    list_page: str = Field(description="The index page listing these vendors.")
    note: str
    cite_as: str = CITE


class Fact(BaseModel):
    value: Any = Field(description="The value as recorded (text, number, list or object).")
    source: str | None = Field(None, description="The vendor's own page the fact was read from.")
    quote: str | None = Field(None, description="The words on that page (emails and phones masked).")
    checked: str | None = None


class VendorDetail(BaseModel):
    name: str
    category: str
    areas: list[str]
    score: int
    price_from_usd: float | None = None
    website: str | None = None
    index_page: str
    checked: str | None = None
    how_to_book: Booking
    facts: dict[str, Fact] = Field(description="Every published fact, by name (starting_price, packages, deposit...).")
    cite_as: str = CITE


class UnitStat(BaseModel):
    unit: str = Field(description='"per hour" or "per guest".')
    vendors: int
    low_quartile: float
    median: float
    high_quartile: float


class PriceSummary(BaseModel):
    category: str
    area: str | None = None
    area_note: str | None = Field(None, description="Set when the area had too few prices and the whole region is used.")
    unit: str | None = None
    as_of: str | None = None
    vendors: int = Field(description="How many vendors' published prices the figures come from.")
    low_quartile: float
    median: float
    high_quartile: float
    per_unit: list[UnitStat] = Field(default_factory=list)
    sentence: str = Field(description="One sentence that states the figures with their source.")
    source: str
    cost_pages: list[dict[str, str]] = Field(default_factory=list,
                                             description="More specific index pages answering what this costs "
                                                         "(wedding, per hour, per guest), each with a one-sentence summary.")
    cite_as: str = CITE


class CategoryRow(BaseModel):
    category: str
    vendors: int
    with_price: int
    median_starting_price_usd: float | None = None
    list_page: str


class CategoryList(BaseModel):
    categories: list[CategoryRow]
    areas: list[str]
    event_types: list[str]
    needs: list[str] = Field(default_factory=list, description="Values the needs parameter accepts.")
    as_of: str | None = None
    cite_as: str = CITE


class ShortlistGroup(BaseModel):
    category: str
    matched: int
    vendors: list[VendorCard]
    list_page: str


class Shortlist(BaseModel):
    event_type: str | None
    area: str | None
    budget_usd: float | None
    guests: int | None = None
    hours: float | None = None
    groups: list[ShortlistGroup]
    how_ranked: str
    cite_as: str = CITE


class BudgetLine(BaseModel):
    category: str
    low_usd: float | None = Field(None, description="25th percentile of published starting prices, for this event.")
    median_usd: float | None = None
    high_usd: float | None = Field(None, description="75th percentile, for this event.")
    basis: str = Field(description='How the figures were worked out, e.g. "median $40 per guest x 120 guests".')
    vendors: int | None = Field(None, description="How many vendors' published prices the figures come from.")
    area: str | None = Field(None, description="Area the figures are for; None means the whole New York area.")
    source: str


class EventBudget(BaseModel):
    area: str | None
    guests: int | None
    hours: float | None
    services: list[BudgetLine]
    total_low_usd: float = Field(description="Sum of each service's 25th percentile: a rough low end, not a quote.")
    total_median_usd: float
    total_high_usd: float = Field(description="Sum of each service's 75th percentile: a rough high end.")
    sentence: str = Field(description="One sentence that states the total with its source.")
    notes: list[str] = Field(default_factory=list)
    as_of: str | None = None
    cite_as: str = CITE


class Comparison(BaseModel):
    vendors: list[VendorDetail]
    side_by_side: list[dict[str, Any]] = Field(description="One row per vendor with the facts most used to choose.")
    cite_as: str = CITE


class SearchHit(BaseModel):
    id: str = Field(description="Pass to fetch.")
    title: str
    url: str


class SearchResults(BaseModel):
    results: list[SearchHit]


class Document(BaseModel):
    id: str
    title: str
    text: str
    url: str
    metadata: dict[str, str] = Field(default_factory=dict)


# ---------- helpers ----------

def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class IndexData:
    """The published files, fetched over HTTP and kept for TTL seconds."""

    def __init__(self, fetch: Callable[[str], Any] | None = None, ttl: float = TTL) -> None:
        self._fetch = fetch or self._http
        self.ttl = ttl
        self._cache: dict[str, tuple[float, Any]] = {}

    @staticmethod
    def _http(path: str) -> Any:
        r = httpx.get(f"{BASE_URL}{path}", timeout=30, follow_redirects=True,
                      headers={"User-Agent": f"runup-index-mcp/{VERSION} (+https://index.therunup.ai/agents/)"})
        r.raise_for_status()
        return r.json()

    def get(self, path: str) -> Any:
        hit = self._cache.get(path)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]
        data = self._fetch(path)
        self._cache[path] = (time.monotonic(), data)
        if len(self._cache) > 300:  # vendor pages are fetched one by one; keep the cache small
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            self._cache.pop(oldest)
        return data

    def vendors(self) -> list[dict]:
        return self.get("/data/vendors.json")

    def prices(self) -> dict:
        return self.get("/prices/index.json")


def category_key(text: str, known: set[str]) -> str | None:
    t = text.strip().lower()
    if _key(t) in known:
        return _key(t)
    if t in CATEGORY_WORDS and CATEGORY_WORDS[t] in known:
        return CATEGORY_WORDS[t]
    for word, key in sorted(CATEGORY_WORDS.items(), key=lambda kv: -len(kv[0])):  # "photo booth" before "photo"
        if key in known and re.search(rf"\b{re.escape(word)}\b", t):
            return key
    return None


BOROUGHS = {"manhattan", "brooklyn", "queens", "bronx", "staten-island"}
NYC_WORDS = ["nyc", "new york city", "the five boroughs", "five boroughs"]
NJ = {"north-jersey", "central-jersey"}  # "new-jersey": both, as "nyc" is the five boroughs
NJ_WORDS = ["new jersey", "nj"]
GROUPS = {"nyc": BOROUGHS, "new-jersey": NJ}


# The site lists a planner only if its own site shows it plans events; officiants and elopement packages keep their
# vendor page (get_vendor finds them by name) but are not offered as planners in searches and shortlists.
LISTED_PLANNER_TYPES = {"full-service planner", "partial planning or day-of coordinator", "event or corporate planner"}


def listed(v: dict) -> bool:
    if v["category"] != "planners":
        return True
    kind = (v["facts"].get("planner_type") or {}).get("value")
    return not isinstance(kind, str) or kind.strip().strip('"').lower() in LISTED_PLANNER_TYPES


def in_area(v: dict, ak: str | None) -> bool:
    """Does the vendor serve the area? "nyc" means any of the five boroughs."""
    if not ak:
        return True
    return bool(GROUPS[ak] & set(v["areas"])) if ak in GROUPS else ak in v["areas"]


NEEDS = {  # what a consumer may need, in plain words, to the yes/no fact the index records
    "kosher": ["kosher"], "halal": ["halal"],
    "mc_services": ["mc", "emcee", "master of ceremonies", "host", "mc services"],
    "second_shooter": ["second shooter", "second photographer", "2 photographers", "two photographers"],
    "second_videographer": ["second videographer", "two videographers"],
    "on_location": ["comes to you", "on location", "travels", "on-site", "onsite", "at home", "at the venue"],
    "staff_included": ["staff", "servers", "waiters", "service staff"],
    "delivery_setup": ["delivery", "setup", "set up", "delivery and setup"],
    "lighting": ["lighting", "lights", "uplighting", "up lighting", "uplights", "dance lights"],
    "photo_booth": ["photo booth", "photobooth"],
}
# needs the free-text search picks out of a question (clear words only; "staff" or "setup" are too common)
SEARCH_NEEDS = ("kosher", "halal", "lighting", "mc_services", "second_shooter")
# needs read from what the vendor's site says it includes or offers, not from a yes/no fact
TEXT_NEEDS = {"lighting": re.compile(r"\blight", re.I), "photo_booth": re.compile(r"photo\s?booth", re.I)}
TEXT_NEED_FIELDS = ("equipment_included", "services", "packages")


def need_keys(words: list[str] | None) -> list[str]:
    keys = []
    for w in words or []:
        t = " ".join(w.lower().replace("_", " ").split())
        k = next((k for k, ws in NEEDS.items() if t == k.replace("_", " ") or t in ws), None)
        if not k:
            raise ToolError(f"Unknown need {w!r}. Use any of: kosher, halal, MC, lighting, photo booth, second "
                            "shooter, second videographer, comes to you, staff, delivery and setup.")
        keys.append(k)
    return list(dict.fromkeys(keys))


def _offers(v: dict, k: str) -> bool | None:
    """True if the vendor's site says it offers k; None if the index records nothing that could say so."""
    if k in TEXT_NEEDS:
        texts = [json.dumps((v["facts"].get(f) or {}).get("value")) for f in TEXT_NEED_FIELDS if f in v["facts"]]
        return any(TEXT_NEEDS[k].search(t) for t in texts) if texts else None
    val = (v["facts"].get(k) or {}).get("value")
    return None if val is None else val is True


def has_needs(v: dict, keys: list[str]) -> bool:
    return all(_offers(v, k) for k in keys)


def area_key(text: str) -> str | None:
    t = text.strip().lower()
    if any(re.search(rf"\b{re.escape(w)}\b", t) for w in NYC_WORDS):
        return "nyc"
    if _key(t) in AREAS:
        return _key(t)
    for key, words in AREAS.items():
        if any(w == t or re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            return key
    if any(re.search(rf"\b{w}\b", t) for w in NJ_WORDS):  # after the counties and towns, which are more precise
        return "new-jersey"
    return None


def event_key(text: str) -> str | None:
    t = text.strip().lower()
    if _key(t) in EVENT_WORDS:
        return _key(t)
    for key, words in EVENT_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            return key
    return None


# Where the site's price_from_usd can come from, in the site's order, with the unit each field implies.
PRICE_SOURCES = (("starting_price", None), ("price_per_hour", "hour"), ("price_per_guest", "guest"),
                 ("price_per_person", "person"), ("price_per_serving", "serving"), ("minimum_order", "event"),
                 ("typical_price_range", None), ("packages", "event"))


def _price(v: dict) -> Price | None:
    """The vendor's published starting price as the site shows it (price_from_usd), with the unit and what it
    covers from the fact it came from: a starting price, a per-hour or per-guest rate, a range or a package."""
    amount = v.get("price_from_usd")
    if amount is None:
        return None
    first = None
    for field, unit in PRICE_SOURCES:
        f = v["facts"].get(field)
        if not f:
            continue
        val = f.get("value")
        if field == "packages":
            prices = [x.get("price") for x in (val or []) if isinstance(x, dict) and isinstance(x.get("price"), (int, float))]
            got, per, note = (min(prices) if prices else None), "event", "lowest listed package"
        elif isinstance(val, dict):
            got, per = val.get("amount", val.get("min")), val.get("per") or unit or "event"
            if got is None and field == "starting_price":  # the site's figure came from it
                got = amount
            note = val.get("note") or ("minimum order" if field == "minimum_order" else None)
        else:
            continue
        if got is None:
            continue
        cand = Price(amount_usd=amount, per=per, covers=note, source=f.get("source"))
        if abs(float(got) - float(amount)) < 0.5:
            return cand
        first = first or cand
    return first


def booking(facts: dict) -> Booking:
    """What the vendor's own site shows for booking, as read on the date given."""
    get = lambda f: (facts.get(f) or {}).get("value")  # noqa: E731
    return Booking(online_booking=get("online_booking"), shows_open_dates=get("shows_open_dates"),
                   instant_quote=get("instant_quote"), quote_request_form=get("quote_request_form"),
                   contact_methods=get("contact_methods") or [])


def vendor_card(v: dict) -> VendorCard:
    return VendorCard(name=v["name"], category=v["category"], areas=v["areas"], score=v["score"],
                      starting_price=_price(v), how_to_book=booking(v["facts"]), website=v.get("website"),
                      index_page=v["url"], checked=v.get("checked"))


def _slug(url: str) -> str | None:
    m = re.search(r"/vendors/([^/]+)/?", url or "")
    return m.group(1) if m else None


def _money(n: float) -> str:
    return f"${n:,.0f}"


def _bookable(v: dict) -> int:
    """How directly a consumer can act on the vendor's site: 3 book online, 2 instant quote or open dates,
    1 quote form, 0 otherwise."""
    f = v["facts"]
    val = lambda k: (f.get(k) or {}).get("value") is True  # noqa: E731
    return 3 if val("online_booking") else 2 if (val("instant_quote") or val("shows_open_dates")) else \
        1 if val("quote_request_form") else 0


# A caterer's or cake maker's price per person is per guest; a hair and makeup artist's is per person made up.
PER_PERSON_IS_PER_GUEST = {"caterers", "cakes-and-desserts"}


def _fit(text: str | None, limit: int) -> str:
    """Whole sentences of text up to limit characters (GPT Actions refuse longer descriptions)."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(out) + len(sent) + 1 > limit:
            break
        out = f"{out} {sent}".strip()
    return out or text[: limit - 1].rsplit(" ", 1)[0] + "."


def estimate(v: dict, guests: int | None, hours: float | None) -> Estimate | None:
    """The published starting price worked out for one event: whole-event prices as they are, per-guest prices times
    the guests, per-hour prices times the hours. None when the rate needs a number the caller did not give."""
    p = _price(v)
    if p is None:
        return None
    if p.per == "event":
        return Estimate(total_usd=p.amount_usd, basis=f"{_money(p.amount_usd)} published for a whole event")
    if guests and (p.per == "guest" or (p.per in ("person", "serving") and v["category"] in PER_PERSON_IS_PER_GUEST)):
        return Estimate(total_usd=round(p.amount_usd * guests, 2),
                        basis=f"{_money(p.amount_usd)} per {p.per} x {guests} guests")
    if p.per == "hour" and hours:
        return Estimate(total_usd=round(p.amount_usd * hours, 2),
                        basis=f"{_money(p.amount_usd)} per hour x {hours:g} hours")
    return None


def log_call(tool: str, args: dict, ctx: Context | None, result: Any, error: str | None = None) -> None:
    """One JSON line per tool call on stdout: tool, arguments, client name and user agent. No IP address."""
    ua = client = None
    try:
        ua = (getattr(ctx, "headers", None) or {}).get("user-agent")
    except Exception:  # noqa: BLE001
        pass
    try:
        info = ctx.session.client_params.client_info  # type: ignore[union-attr]
        client = f"{info.name}/{info.version}"
    except Exception:  # noqa: BLE001
        pass
    r = result.model_dump() if isinstance(result, BaseModel) else (result if isinstance(result, dict) else {})
    matched = r.get("matched")
    if matched is None and isinstance(r.get("results"), list):
        matched = len(r["results"])
    line = {"mcp_call": tool, "args": {k: v for k, v in args.items() if v not in (None, False, "", [])},
            "user_agent": ua, "client": client, "error": error, "matched": matched, "vendor": r.get("name")}
    print(json.dumps(line, default=str), file=sys.stdout, flush=True)


def logged(fn):
    """Log every call of a tool (see log_call), errors included, without changing what it returns."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        call_args = {k: v for k, v in kwargs.items() if k != "ctx"}
        try:
            out = fn(*args, **kwargs)
        except ToolError as e:
            log_call(fn.__name__, call_args, kwargs.get("ctx"), None, str(e))
            raise
        log_call(fn.__name__, call_args, kwargs.get("ctx"), out)
        return out
    return wrapper


# ---------- parameters ----------

CategoryArg = Annotated[str, Field(description='Vendor category in plain words: "photographers", "DJ", "caterer", '
                                               '"florist", "planner", "live band", "videographer", "photo booth", '
                                               '"hair and makeup" or "cakes". list_categories gives the full list.')]
AreaArg = Annotated[str | None, Field(description="Optional area: NYC (any of the five boroughs), Manhattan, Brooklyn, "
                                                  "Queens, Bronx, Staten Island, Long Island, Westchester, North Jersey or "
                                                  "Central Jersey, or New Jersey for both; a county or town also works (Bergen County, "
                                                  "Hoboken, Monmouth County). Omit for the whole New York area.")]
EventArg = Annotated[str | None, Field(description='Optional event type in plain words: "wedding", "bar mitzvah", '
                                                   '"sweet 16", "birthday", "corporate holiday party", "baby shower".')]


NeedsArg = Annotated[list[str] | None, Field(description='Optional things the vendor must offer, as its own website '
                                                       'states: "kosher", "halal" (caterers), "MC", "lighting", '
                                                       '"photo booth" (DJs, bands), "second shooter" (photographers), '
                                                       '"second videographer", '
                                                       '"comes to you" (hair and makeup), "staff" (caterers), '
                                                       '"delivery and setup" (florists, cakes).')]


def build_server(data: IndexData | None = None) -> MCPServer:
    data = data or IndexData()
    server = MCPServer(
        name="agent-bookability-index",
        title="Agent Bookability Index",
        website_url=f"{BASE_URL}/agents/",
        icons=[Icon(src=f"{BASE_URL}/favicon.svg", mimeType="image/svg+xml")],
        version=VERSION,
        instructions=(
            "Published facts about New York area private-event vendors (photographers, DJs, caterers, florists, "
            "planners, videographers, live bands, photo booths, hair and makeup, cakes): starting prices with what "
            "they cover, areas served, event types, and how to book, each read from the vendor's own website with "
            "the page it came from. Also median prices by category and area.\n"
            "Typical flow: shortlist_vendors for an event (or search_vendors for one category), then get_vendor or "
            "compare_vendors for details, get_price_summary for the published starting prices of a service, and "
            "estimate_event_budget for the published starting prices of all the services together.\n"
            "Areas: a borough, NYC, Long Island, Westchester, North or Central Jersey, New Jersey, or a county or town "
            "in them (Bergen County, Hoboken). Needs: kosher, halal, MC, lighting, photo booth, second shooter and more "
            "(list_categories lists them). For the cheapest options, search_vendors with sort=price; mind each price's "
            "unit (per event, hour or guest).\n"
            "When you use a fact, cite the index_page or source given and say the date it was checked. Prices are "
            "what vendors publish, not quotes; the consumer confirms with the vendor using the booking route shown. "
            "The score measures how much an AI agent can learn and do on a vendor's own site, not the quality of "
            "the vendor's work. Only some vendors publish prices, so price figures describe those vendors."),
    )

    @server.custom_route("/.well-known/mcp-registry-auth", methods=["GET"], include_in_schema=False)
    async def registry_auth(request):  # noqa: ARG001
        return PlainTextResponse(REGISTRY_AUTH + "\n")

    @server.custom_route("/.well-known/glama.json", methods=["GET"], include_in_schema=False)
    async def glama(request):  # noqa: ARG001
        return JSONResponse({"$schema": "https://glama.ai/mcp/schemas/server.json"})

    @server.custom_route("/mcp/.well-known/glama.json", methods=["GET"], include_in_schema=False)
    async def glama_under_mcp(request):  # noqa: ARG001
        return await glama(request)

    @server.custom_route("/.well-known/openai-apps", methods=["GET"], include_in_schema=False)
    async def openai_apps(request):  # noqa: ARG001
        token = os.environ.get("OPENAI_APPS_TOKEN", "").strip()
        return PlainTextResponse(token + "\n") if token else PlainTextResponse("not set\n", status_code=404)

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(request):  # noqa: ARG001
        return PlainTextResponse("ok\n")

    def ro(title: str) -> ToolAnnotations:
        return ToolAnnotations(title=title, readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                               openWorldHint=False)

    def known_categories() -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in data.vendors():
            counts[v["category"]] = counts.get(v["category"], 0) + 1
        return counts

    def need_category(text: str) -> str:
        counts = known_categories()
        cat = category_key(text, set(counts))
        if not cat and re.search(r"\b(venues?|halls?|ballrooms?|banquet|reception site|space|spaces)\b", text, re.I):
            raise ToolError("The index covers event vendors, not venues. For the services at a venue use one of: "
                            f"{', '.join(sorted(counts))}.")
        if not cat:
            raise ToolError(f"Unknown category {text!r}. Use one of: {', '.join(sorted(counts))}.")
        return cat

    def need_area(text: str | None) -> str | None:
        if not text:
            return None
        ak = area_key(text)
        if not ak:
            raise ToolError(f"Unknown area {text!r}. The index covers the New York area only. Use NYC, New Jersey or "
                            f"one of: {', '.join(sorted(AREAS))}, or omit it.")
        return ak

    def need_event(text: str | None) -> str | None:
        if not text:
            return None
        ek = event_key(text)
        if not ek:
            raise ToolError(f"Unknown event type {text!r}. Use one of: {', '.join(EVENT_WORDS)}, or omit it.")
        return ek

    def serves_event(v: dict, ek: str | None) -> bool:
        if not ek:
            return True
        types = (v["facts"].get("event_types") or {}).get("value") or []
        return ek in types or (ek != "weddings" and "other-private" in types)

    def vendor_page(ref: str) -> dict:
        """The vendor's full page JSON from a name, index_page URL or slug."""
        slug = _slug(ref)
        if not slug:
            want = ref.strip().lower()
            vs = data.vendors()
            exact = [v for v in vs if v["name"].lower() == want]
            close = exact or [v for v in vs if want in v["name"].lower()]
            if len(close) > 1 and not exact:
                listing = "; ".join(f"{v['name']} ({v['category']}): {v['url']}" for v in close[:10])
                raise ToolError(f"{len(close)} vendors match {ref!r}. Pass one index_page: {listing}")
            slug = _slug(close[0]["url"]) if close else _key(ref)
        try:
            return data.get(f"/vendors/{slug}/index.json")
        except httpx.HTTPStatusError:
            raise ToolError(f"No vendor {ref!r} in the index. Try search_vendors with name_contains.") from None

    def detail(page: dict) -> VendorDetail:
        facts = {k: Fact(value=f.get("value"), source=f.get("source"), quote=f.get("quote"), checked=f.get("checked"))
                 for k, f in (page.get("facts") or {}).items()}
        return VendorDetail(name=page["name"], category=page["category"], areas=page.get("areas") or [],
                            score=page.get("score") or 0, price_from_usd=page.get("price_from_usd"),
                            website=page.get("website"), index_page=page.get("url") or f"{BASE_URL}{page.get('path', '')}",
                            checked=page.get("checked"), how_to_book=booking(page.get("facts") or {}), facts=facts)

    def price_summary(category: str, area: str | None) -> PriceSummary:
        prices = data.prices()
        rows = prices.get("by_category", [])
        known = {r["key"] for r in rows} | set(known_categories())
        cat = category_key(category, known)
        row = next((r for r in rows if r["key"] == cat), None)
        if not row:
            raise ToolError(f"Not enough published prices for {category!r} yet. Categories with prices: "
                            f"{', '.join(r['key'] for r in rows)}.")
        label = row["category"]
        said = label if label[1:2].isupper() else label[:1].lower() + label[1:]  # "DJs" stays, "Florists" -> "florists"
        month = time.strftime("%B %Y", time.strptime(prices["updated"], "%Y-%m-%d"))
        where, stats, ak, note = "the New York area", row, None, None
        if area:
            want = area_key(area)
            arow = next((r for r in prices.get("by_area", [])
                         if r["category"] == label and want and area_key(r["area"]) == want), None)
            if arow:
                where, stats, ak = arow["area"], arow, want
            elif want == "nyc":
                note = "Prices are split by borough, not for NYC as one; showing the whole New York area. Ask per borough."
            elif want == "new-jersey":
                note = ("Prices are split into North Jersey and Central Jersey; showing the whole New York area. Ask for "
                        "North Jersey or Central Jersey.")
            else:
                note = f"Not enough prices for {area!r}; showing the whole New York area."
        units = [UnitStat(unit=u["unit"], vendors=u["vendors"], low_quartile=u["low_quartile"], median=u["median"],
                          high_quartile=u["high_quartile"])
                 for u in prices.get("per_unit", []) if u["category"] == label]
        sentence = (f"The median starting price {said} in {where} publish for a whole event is "
                    f"{_money(stats['median'])}, with the middle half between {_money(stats['low_quartile'])} "
                    f"and {_money(stats['high_quartile'])} ({stats['vendors']} vendors' own published prices, "
                    f"{month}, Agent Bookability Index).")
        for u in units:  # caterers, photographers by the hour and others are often priced per unit, not per event
            sentence += (f" Priced {u.unit}, the median is {_money(u.median)}, with the middle half between "
                         f"{_money(u.low_quartile)} and {_money(u.high_quartile)} ({u.vendors} vendors).")
        asked = area_key(area) if area else None  # the area asked for, even when the price index has no row for it
        pages = []
        try:
            for q in (data.get("/cost/index.json") or {}).get("questions", []):
                if q.get("category") == cat and q.get("area") in (None, asked):
                    pages.append({"question": q["question"], "answer": q["answer"], "url": q["url"],
                                  "_area": q.get("area")})
        except httpx.HTTPError:
            pass
        pages.sort(key=lambda p: (0 if asked and p["_area"] == asked else 1, p["question"]))
        pages = [{k: v for k, v in p.items() if k != "_area"} for p in pages]
        return PriceSummary(cost_pages=pages[:4], category=cat, area=ak, area_note=note, unit=prices.get("unit"), as_of=prices.get("updated"),
                            vendors=stats["vendors"], low_quartile=stats["low_quartile"], median=stats["median"],
                            high_quartile=stats["high_quartile"], per_unit=units, sentence=sentence,
                            source=f"{BASE_URL}/prices/")

    def rank(vs: list[dict], budget: float | None, guests: int | None = None,
             hours: float | None = None) -> list[dict]:
        """Order: within budget (when one is given), can be booked or quoted on the site, publishes a price,
        then score, then price. A per-guest or per-hour price is compared with the budget only once it can be
        worked out for the event."""
        def key(v):
            p = v.get("price_from_usd")
            est = estimate(v, guests, hours)
            fits = 1 if budget is None or (est is not None and est.total_usd <= budget) else 0
            return (-fits, -_bookable(v), 0 if p is not None else 1, -v["score"],
                    est.total_usd if est else p if p is not None else float("inf"), v["name"])
        return sorted(vs, key=key)

    # ---------- tools ----------

    @server.tool(title="List vendor categories", annotations=ro("List vendor categories"))
    @logged
    def list_categories(ctx: Context | None = None) -> CategoryList:
        """What the index covers: the vendor categories with how many vendors each has, how many publish a price and
        the median published starting price for a whole event; the areas, event types and needs (kosher, MC,
        lighting...) the other tools accept. Call it first if you are unsure what the index covers."""
        counts = known_categories()
        priced: dict[str, int] = {}
        for v in data.vendors():
            if _price(v):
                priced[v["category"]] = priced.get(v["category"], 0) + 1
        medians = {c["key"]: c["median"] for c in data.prices().get("by_category", [])}
        rows = [CategoryRow(category=k, vendors=n, with_price=priced.get(k, 0), median_starting_price_usd=medians.get(k),
                            list_page=f"{BASE_URL}/nyc/{k}/") for k, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        return CategoryList(categories=rows, areas=["nyc", "new-jersey", *sorted(AREAS)], event_types=list(EVENT_WORDS),
                            needs=[k.replace("_", " ") for k in NEEDS],
                            as_of=data.prices().get("updated"))

    @server.tool(title="Search vendors", annotations=ro("Search vendors"))
    @logged
    def search_vendors(
            category: Annotated[str | None, Field(description='Vendor category in plain words ("DJ", "wedding '
                                                  'photographer", "florist"). Omit to search every category.')] = None,
            area: AreaArg = None,
            event_type: EventArg = None,
            max_price_usd: Annotated[float | None, Field(description="Only vendors whose published starting price "
                                                         "is at or under this, in US dollars.")] = None,
            has_price: Annotated[bool, Field(description="Only vendors that publish a price.")] = False,
            books_online: Annotated[bool, Field(description="Only vendors whose own site shows online booking.")] = False,
            name_contains: Annotated[str | None, Field(description="Part of the vendor's name.")] = None,
            needs: NeedsArg = None,
            limit: Annotated[int, Field(description="How many vendors to return, 1 to 25 (larger values are capped).")] = 10,
            offset: Annotated[int, Field(description="How many matches to skip, to page through a long list: pass "
                                                     "next_offset from the previous result.")] = 0,
            sort: Annotated[str, Field(description='"score" (default: most bookable first, then price) or "price" '
                                                   '(lowest published starting price first, vendors without a price '
                                                   'last; check starting_price.per, as a per-hour or per-guest rate '
                                                   'is not a whole-event price).')] = "score",
            ctx: Context | None = None) -> SearchVendorsResult:
        """Find New York area event vendors, in one category or across all, filtered by area, event type, maximum
        published price, online booking or part of a name. Sorted by bookability score (how much an AI agent can
        learn and do on the vendor's own site), then price. Each result has the starting price and what it covers,
        how to book, and the index page to cite. For a ready shortlist across several categories, use
        shortlist_vendors; search is the id-only variant for ChatGPT."""
        cat = need_category(category) if category else None
        ak, ek = need_area(area), need_event(event_type)
        limit = max(1, min(int(limit), MAX_LIMIT))
        needle = (name_contains or "").strip().lower()
        nk = need_keys(needs)
        hits = []
        for v in data.vendors():
            if (cat and v["category"] != cat) or not listed(v) or not in_area(v, ak) or not serves_event(v, ek):
                continue
            if needle and needle not in v["name"].lower():
                continue
            if nk and not has_needs(v, nk):
                continue
            p = v.get("price_from_usd")
            if (has_price or max_price_usd is not None) and not _price(v):
                continue
            if max_price_usd is not None and p is not None and p > max_price_usd:
                continue
            if books_online and (v["facts"].get("online_booking") or {}).get("value") is not True:
                continue
            hits.append(v)
        if sort not in ("score", "price"):
            raise ToolError('sort must be "score" or "price".')
        if sort == "price":
            hits.sort(key=lambda v: (v.get("price_from_usd") if v.get("price_from_usd") is not None else float("inf"),
                                     -v["score"], v["name"]))
        else:
            hits.sort(key=lambda v: (-v["score"], v.get("price_from_usd") or float("inf"), v["name"]))
        page = (f"{BASE_URL}/nyc/{cat}/" + (f"{ak}/" if ak and ak not in GROUPS else "")) if cat else f"{BASE_URL}/"
        start = max(0, int(offset or 0))
        shown = hits[start:start + limit]
        nxt = start + limit if start + limit < len(hits) else None
        return SearchVendorsResult(category=cat, area=ak, matched=len(hits), shown=len(shown), offset=start,
                                   next_offset=nxt, vendors=[vendor_card(v) for v in shown], list_page=page,
                                   note="Prices and booking options are what each vendor's own site showed on the "
                                        "checked date.")

    @server.tool(title="Shortlist vendors for an event", annotations=ro("Shortlist vendors for an event"))
    @logged
    def shortlist_vendors(
            categories: Annotated[list[str], Field(description='The services needed, in plain words, e.g. ["DJ", '
                                                   '"photographer", "florist"]. Up to 6.')],
            event_type: EventArg = None,
            area: AreaArg = None,
            budget_usd: Annotated[float | None, Field(description="Optional budget per service, US dollars. Vendors "
                                                      "whose published starting price fits come first.")] = None,
            per_category: Annotated[int, Field(description="Vendors per service, 1 to 10.")] = 3,
            needs: NeedsArg = None,
            guests: Annotated[int | None, Field(description="Optional number of guests. Per-guest prices (catering, "
                                                            "cakes) are worked out for this many.")] = None,
            hours: Annotated[float | None, Field(description="Optional length of the event in hours. Per-hour prices "
                                                             "(some DJs, photographers, bands) are worked out for "
                                                             "this long.")] = None,
            ctx: Context | None = None) -> Shortlist:
        """A shortlist for one event: for each service asked for, the most bookable New York area vendors that serve
        the area and event type. Ranked by how directly a consumer can act (book online, instant quote or open
        dates, quote form), then by fit with the budget, bookability score and price. Pass guests and hours to get each
        vendor's published rate worked out for the event (estimate). Use it to answer "help me find a DJ, caterer and
        florist for a 120-guest bar mitzvah in Brooklyn, 4 hours, under $3,000 each"."""
        if not categories:
            raise ToolError("Pass at least one category, e.g. [\"DJ\"].")
        cats = [need_category(c) for c in categories[:6]]
        ak, ek = need_area(area), need_event(event_type)
        n = max(1, min(int(per_category), 10))
        g = int(guests) if guests and guests > 0 else None
        h = float(hours) if hours and hours > 0 else None
        nk = need_keys(needs)
        groups = []
        for cat in dict.fromkeys(cats):
            # a need applies only to the services it makes sense for (kosher to caterers, MC to DJs and bands)
            want = [k for k in nk if any(_offers(v, k) for v in data.vendors() if v["category"] == cat)]
            pool = [v for v in data.vendors() if v["category"] == cat and listed(v) and in_area(v, ak) and serves_event(v, ek)
                    and has_needs(v, want)]
            ranked = rank(pool, budget_usd, g, h)
            cards = [vendor_card(v).model_copy(update={"estimate": estimate(v, g, h)}) for v in ranked[:n]]
            groups.append(ShortlistGroup(category=cat, matched=len(pool), vendors=cards,
                                         list_page=f"{BASE_URL}/nyc/{cat}/" + (f"{ak}/" if ak and ak not in GROUPS else "")))
        return Shortlist(event_type=ek, area=ak, budget_usd=budget_usd, guests=g, hours=h, groups=groups,
                         how_ranked=("Within budget first, then " if budget_usd is not None else "") +
                                    "vendors you can book or get a price from on their own site, then vendors that "
                                    "publish a price, then bookability score, then lowest price for this event.")

    @server.tool(title="Get one vendor", annotations=ro("Get one vendor"))
    @logged
    def get_vendor(vendor: Annotated[str, Field(description="The vendor's name (or part of it), its index_page URL "
                                                        "from another tool, or its slug such as \"448-studio\".")],
                   ctx: Context | None = None) -> VendorDetail:
        """Everything the index knows about one vendor: category, areas, bookability score, website, how to book,
        and every published fact (starting price and what it covers, packages, deposit, booking lead time,
        cancellation policy and more), each with the quote from the vendor's own site, its page and the date read.
        If a name matches several vendors, the error lists them with their index_page. (fetch returns the same facts
        as text, for ids that came from search.)"""
        return detail(vendor_page(vendor))

    @server.tool(title="Compare vendors", annotations=ro("Compare vendors"))
    @logged
    def compare_vendors(
            vendors: Annotated[list[str], Field(description="2 to 5 vendors, each a name, index_page URL or slug.")],
            ctx: Context | None = None) -> Comparison:
        """Two to five vendors side by side: starting price and what it covers, areas, event types, deposit, booking
        lead time, cancellation policy and how to book, with every fact's source. Use it when a consumer is choosing
        between vendors from a search or shortlist."""
        if not 2 <= len(vendors) <= 5:
            raise ToolError("Pass between 2 and 5 vendors.")
        details = [detail(vendor_page(v)) for v in vendors]
        keys = ("starting_price", "typical_price_range", "deposit", "booking_lead_time", "cancellation_policy",
                "event_types", "hours_included")
        rows = []
        for d in details:
            row: dict[str, Any] = {"name": d.name, "category": d.category, "score": d.score,
                                   "price_from_usd": d.price_from_usd, "index_page": d.index_page,
                                   "books_online": d.how_to_book.online_booking,
                                   "quote_form": d.how_to_book.quote_request_form}
            row.update({k: d.facts[k].value for k in keys if k in d.facts})
            rows.append(row)
        return Comparison(vendors=details, side_by_side=rows)

    @server.tool(title="Price summary for a service", annotations=ro("Price summary for a service"))
    @logged
    def get_price_summary(category: CategoryArg, area: AreaArg = None, ctx: Context | None = None) -> PriceSummary:
        """One service's price. What New York area vendors in one category publish as a starting price for a whole event: the median and
        the middle half (25th to 75th percentile), from how many vendors and when, optionally for one area, plus
        per-hour or per-guest figures where the category is priced that way. Comes with a one-sentence summary
        and the page to cite. Use it to answer "how much does a wedding DJ cost in Brooklyn"."""
        return price_summary(category, area)

    @server.tool(title="Budget for an event", annotations=ro("Budget for an event"))
    @logged
    def estimate_event_budget(
            categories: Annotated[list[str], Field(description='The services the event needs, in plain words, e.g. '
                                                   '["DJ", "photographer", "caterer", "cake"]. Up to 8.')],
            area: AreaArg = None,
            guests: Annotated[int | None, Field(description="Number of guests. Catering and cakes, usually priced "
                                                            "per guest, are worked out for this many.")] = None,
            hours: Annotated[float | None, Field(description="Length of the event in hours. Where a service has "
                                                             "per-hour prices, they are worked out for this long.")] = None,
            ctx: Context | None = None) -> EventBudget:
        """Several services together. The published starting prices of a whole event's vendors: for each service, the low, median and high published
        starting price in the area, worked out for the guests and hours given, and the sum. Use it to answer "what
        will a DJ, photographer and caterer cost for a 120-guest bar mitzvah in Brooklyn". For one service in
        detail use get_price_summary; for actual vendors use shortlist_vendors."""
        if not categories:
            raise ToolError('Pass at least one service, e.g. ["DJ", "caterer"].')
        g = int(guests) if guests and guests > 0 else None
        h = float(hours) if hours and hours > 0 else None
        lines: list[BudgetLine] = []
        notes: list[str] = []
        as_of = None
        for cat in dict.fromkeys(need_category(c) for c in categories[:8]):
            try:
                ps = price_summary(cat, area)
            except ToolError:
                notes.append(f"{cat}: too few published prices to estimate.")
                continue
            as_of = ps.as_of
            if ps.area_note:
                notes.append(f"{cat}: {ps.area_note}")
            units = {u.unit: u for u in ps.per_unit}
            per_guest = units.get("per guest") or units.get("per serving")
            if per_guest and g:
                n, u = g, per_guest
            elif "per hour" in units and h:
                n, u = h, units["per hour"]
            else:
                n, u = None, None
            if u is not None:
                lines.append(BudgetLine(category=cat, low_usd=round(u.low_quartile * n), median_usd=round(u.median * n),
                                        high_usd=round(u.high_quartile * n), vendors=u.vendors, area=None,
                                        basis=f"median {_money(u.median)} {u.unit} x {n:g} "
                                              f"{'guests' if u.unit != 'per hour' else 'hours'}, New York area",
                                        source=ps.source))
            else:
                if cat in PER_PERSON_IS_PER_GUEST and not g:
                    notes.append(f"{cat}: usually priced per guest; pass guests for a better figure.")
                lines.append(BudgetLine(category=cat, low_usd=ps.low_quartile, median_usd=ps.median,
                                        high_usd=ps.high_quartile, vendors=ps.vendors, area=ps.area,
                                        basis="published starting price for a whole event", source=ps.source))
        if not lines:
            raise ToolError("None of these services has enough published prices yet. " + " ".join(notes))
        lo = sum(x.low_usd or 0 for x in lines)
        med = sum(x.median_usd or 0 for x in lines)
        hi = sum(x.high_usd or 0 for x in lines)
        said = [{"djs": "DJs"}.get(x.category, x.category.replace("-", " ")) for x in lines]
        names = said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]
        ak = area_key(area) if area else None
        place = {"nyc": "NYC", "bronx": "the Bronx"}.get(ak or "", (ak or area or "").replace("-", " ").title())
        where = f" in {place}" if area else " in the New York area"
        who = f" for {g} guests" if g else ""
        sentence = (f"Published starting prices for {names}{where}{who} add up to about {_money(med)} at the median, "
                    f"roughly {_money(lo)} to {_money(hi)} (Agent Bookability Index, from vendors' own websites; "
                    "starting prices, not quotes).")
        return EventBudget(area=area_key(area) if area else None, guests=g, hours=h, services=lines,
                           total_low_usd=lo, total_median_usd=med, total_high_usd=hi, sentence=sentence, notes=notes,
                           as_of=as_of)

    # ChatGPT connectors and deep research call these two names with these shapes.
    @server.tool(title="Search the index (ChatGPT-compatible)", annotations=ro("Search the index (ChatGPT-compatible)"))
    @logged
    def search(query: Annotated[str, Field(description='What to look for in plain words, e.g. "wedding DJ Brooklyn '
                                                       'under 2000" or "how much does a florist cost in Queens".')],
               ctx: Context | None = None) -> SearchResults:
        """For ChatGPT connectors and deep research only; other clients use search_vendors. Free-text search in the id-only shape they require: returns ids,
        titles and URLs of vendors, price summaries and cost pages, to open with fetch. Words for a service, an area, an
        event type and a dollar amount narrow the results. Other clients get full results in one call from
        search_vendors (structured filters, vendor cards) or get_price_summary (what a service costs)."""
        q = query.lower()
        known = set(known_categories())
        cat = category_key(q, known)
        ak = area_key(q)
        ek = event_key(q)
        m = re.search(r"(?:\$|under|below|less than|max(?:imum)?|budget(?: of)?|up to)\s*\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?\b",
                      q)
        cap = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1) if m else None
        hits: list[SearchHit] = []
        if cat and (re.search(r"cost|price|how much|budget|typical|average|median", q) or not cap):
            try:
                ps = price_summary(cat, ak)
                label = next((r["category"] for r in data.prices().get("by_category", []) if r["key"] == cat), cat)
                said = label if label[1:2].isupper() else label[:1].lower() + label[1:]
                where = {"bronx": "the Bronx", None: "the New York area"}.get(ps.area, (ps.area or "").replace("-", " ").title())
                hits.append(SearchHit(id=f"price:{cat}:{ps.area or 'all'}", url=ps.source, title=f"What {said} cost in {where}"))
                for cp in ps.cost_pages[:2]:
                    hits.append(SearchHit(id="cost:" + cp["url"].split("/cost/", 1)[1].strip("/"), url=cp["url"],
                                          title=cp["question"]))
            except ToolError:
                pass
        generic = {w for k in CATEGORY_WORDS for w in k.split()} | set(known) | {w for ws in AREAS.values() for p in ws
                                                                               for w in p.split()} \
            | {w for ws in EVENT_WORDS.values() for p in ws for w in p.split()} | {w for p in NYC_WORDS for w in p.split()} \
            | STOP_WORDS
        # needs named in the question ("kosher caterer", "DJ with uplighting and MC") filter like search_vendors' needs
        qn = [k for k in SEARCH_NEEDS if any(re.search(rf"\b{re.escape(w)}\b", q) for w in NEEDS[k])]
        generic |= {w for k in qn for p in NEEDS[k] for w in p.split()}
        applies = {(c, k) for k in qn for c in known if any(_offers(x, k) for x in data.vendors() if x["category"] == c)}
        words = [w for w in re.findall(r"[a-z0-9']+", q) if len(w) > 3 and w not in generic and not w.isdigit()]
        pool = []
        for v in data.vendors():
            if (cat and v["category"] != cat) or not listed(v) or not in_area(v, ak) or not serves_event(v, ek):
                continue
            if qn and not has_needs(v, [k for k in qn if (v["category"], k) in applies]):
                continue
            if qn and not cat and not any((v["category"], k) in applies for k in qn):
                continue
            p = v.get("price_from_usd")
            if cap is not None and (p is None or p > cap):
                continue
            if not (cat or ak or ek or cap or qn) and not any(w in v["name"].lower() for w in words):
                continue
            pool.append(v)
        named = [v for v in pool if any(re.search(rf"\b{re.escape(w)}", v["name"].lower()) for w in words)]
        for v in (named or rank(pool, cap))[:10]:
            price = f", from {_money(v['price_from_usd'])}" if v.get("price_from_usd") else ""
            hits.append(SearchHit(id=f"vendor:{_slug(v['url'])}", url=v["url"],
                                  title=f"{v['name']} ({v['category'].replace('-', ' ')}{price})"))
        return SearchResults(results=hits)

    @server.tool(title="Fetch a result (ChatGPT-compatible)", annotations=ro("Fetch a result (ChatGPT-compatible)"))
    @logged
    def fetch(id: Annotated[str, Field(description='An id from search, e.g. "vendor:448-studio" or '
                                                   '"price:djs:brooklyn".')],
              ctx: Context | None = None) -> Document:
        """For ChatGPT connectors and deep research only; other clients use get_vendor. Opens one id returned by search (a vendor, a price summary or a cost page) as plain text with the URL to
        cite, in the shape ChatGPT expects. To look up a vendor by name, URL or slug without searching first, use
        get_vendor, which returns the same facts as structured fields."""
        kind, _, rest = id.partition(":")
        if kind == "price":
            cat, _, area = rest.partition(":")
            ps = price_summary(cat, None if area in ("", "all") else area)
            lines = [ps.sentence] + [f"Per {u.unit.replace('per ', '')}: median {_money(u.median)} "
                                     f"({_money(u.low_quartile)} to {_money(u.high_quartile)}, {u.vendors} vendors)."
                                     for u in ps.per_unit]
            return Document(id=id, title=f"What {ps.category.replace('-', ' ')} cost", text="\n".join(lines),
                            url=ps.source, metadata={"as_of": ps.as_of or "", "vendors": str(ps.vendors)})
        if kind == "cost" and rest:
            for q in (data.get("/cost/index.json") or {}).get("questions", []):
                if q["url"].rstrip("/").endswith("/cost/" + rest.strip("/")):
                    st = q.get("price_stats_usd") or {}
                    return Document(id=id, title=q["question"], text=q["answer"], url=q["url"],
                                    metadata={k: str(v) for k, v in st.items()})
            raise ToolError(f"No cost page {rest!r}. Use an id from search.")
        if kind != "vendor" or not rest:
            raise ToolError('Unknown id. Use an id from search, like "vendor:448-studio" or "price:djs:brooklyn".')
        d = detail(vendor_page(rest))
        lines = [f"{d.name}: {d.category.replace('-', ' ')} serving {', '.join(d.areas)}. Bookability score "
                 f"{d.score}/100. Website: {d.website}. Checked {d.checked}."]
        for k, f in d.facts.items():
            val = f.value if isinstance(f.value, str) else json.dumps(f.value)
            lines.append(f"{k.replace('_', ' ')}: {val} (source: {f.source})")
        return Document(id=id, title=d.name, text="\n".join(lines), url=d.index_page,
                        metadata={"category": d.category, "checked": d.checked or "", "score": str(d.score)})

    # ---------- prompts and resources ----------

    @server.prompt(title="Find vendors for an event")
    def plan_event_vendors(event: str, area: str = "", services: str = "DJ, photographer", budget: str = "") -> str:
        """A ready request that walks an agent through shortlisting, comparing and pricing vendors for one event."""
        return (f"Help me book vendors for a {event}" + (f" in {area}" if area else "") + f". I need: {services}."
                + (f" Budget per service: {budget}." if budget else "") + " Use shortlist_vendors first (with the "
                "guests, hours and any needs such as kosher catering or an MC), then estimate_event_budget to tell me "
                "the published starting prices of the services together, and compare_vendors for the first few. "
                "For each vendor say its starting price and what it covers, how to book, and cite its index_page "
                "with the date it was checked.")

    @server.resource("index://about", name="about", title="About the Agent Bookability Index", mime_type="text/plain")
    def about() -> str:
        """What the index is, how facts are collected, and how to cite it."""
        return ("The Agent Bookability Index (https://index.therunup.ai/) by The Runup reads each New York area "
                "event vendor's own website and records what it publishes: prices and what they cover, areas, event "
                "types and how to book, with the page and date of every fact. Facts are read by software from vendors' own pages and can contain errors. "
                "Scores say how much an AI agent can learn and do on the vendor's own site, not the quality of the "
                "vendor's work. The Runup, which runs this index, also builds software for event vendors; the same "
                "rules and scoring apply to every vendor, and no vendor can pay for a higher score or a better place "
                "in a list. Method: "
                "https://index.therunup.ai/method/ . Data: https://index.therunup.ai/data/ (our compilation under CC BY 4.0; quotes from vendors' own sites remain theirs). Cite as: "
                + CITE + ".")

    # ---------- plain HTTP API ----------
    # The same tools as GET requests with query parameters, for agents without MCP: GPT Actions and other OpenAPI
    # clients, and any assistant whose browsing tool can open a URL. /openapi.json describes them from the tools' own
    # schemas, so the two never drift apart. Calls are logged like MCP calls, with client "http-api".

    api_tools: dict[str, Callable] = {f.__name__: f for f in (
        list_categories, search_vendors, shortlist_vendors, get_vendor, compare_vendors, get_price_summary,
        estimate_event_budget)}
    # Tool calls are never cached, so answers are always current. The spec and the server card may be cached.
    api_headers = {"Access-Control-Allow-Origin": "*", "Cache-Control": "private, no-store"}
    doc_headers = {"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=300"}

    def api_args(name: str, schema: dict, query) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, prop in (schema.get("properties") or {}).items():
            vals = query.getlist(key)
            if not vals:
                continue
            types = {t.get("type") for t in prop.get("anyOf", [prop])}
            try:
                if "array" in types:
                    out[key] = [x.strip() for v in vals for x in v.split(",") if x.strip()]
                elif "boolean" in types:
                    out[key] = vals[0].lower() in ("1", "true", "yes", "on")
                elif "integer" in types:
                    out[key] = int(float(vals[0]))
                elif "number" in types:
                    out[key] = float(vals[0].replace(",", "").lstrip("$"))
                else:
                    out[key] = vals[0]
            except ValueError as e:
                raise ToolError(f"{key} must be a number, got {vals[0]!r}.") from e
        missing = [k for k in schema.get("required", []) if k not in out]
        if missing:
            raise ToolError(f"{name} needs {', '.join(missing)}. See {BASE_URL}/openapi.json.")
        return out

    async def tool_schemas() -> dict[str, Any]:
        return {t.name: t for t in await server.list_tools() if t.name in api_tools}

    @server.custom_route("/api/{tool}", methods=["GET"], include_in_schema=False)
    async def api_call(request):
        name = request.path_params["tool"].replace("-", "_")
        tools = await tool_schemas()
        if name not in tools:
            return JSONResponse({"error": f"No tool {name!r}. Use one of: {', '.join(api_tools)}.",
                                 "openapi": f"{BASE_URL}/openapi.json"}, status_code=404, headers=api_headers)
        t = tools[name]
        ctx = SimpleNamespace(headers=request.headers, session=SimpleNamespace(client_params=SimpleNamespace(
            client_info=SimpleNamespace(name="http-api", version=VERSION))))
        try:
            args = api_args(name, t.input_schema, request.query_params)
            result = api_tools[name](**args, ctx=ctx)
        except ToolError as e:
            return JSONResponse({"error": str(e).removeprefix(f"Error executing tool {name}: ")}, status_code=400,
                                headers=api_headers)
        return JSONResponse(result.model_dump(mode="json"), headers=api_headers)

    @server.custom_route("/api/{tool}/", methods=["GET"], include_in_schema=False)
    async def api_call_slash(request):
        return await api_call(request)

    @server.custom_route("/api/", methods=["GET"], include_in_schema=False)
    async def api_index_slash(request):
        return await api_index(request)

    @server.custom_route("/api", methods=["GET"], include_in_schema=False)
    async def api_index(request):  # noqa: ARG001
        tools = await tool_schemas()
        return JSONResponse({"about": "Agent Bookability Index: the MCP server's tools as plain GET requests. No key.",
                             "openapi": f"{BASE_URL}/openapi.json", "mcp": f"{BASE_URL}/mcp",
                             "tools": {n: {"url": f"{BASE_URL}/api/{n}", "description": t.description}
                                       for n, t in tools.items()},
                             "example": f"{BASE_URL}/api/shortlist_vendors?categories=DJ,caterer&event_type=bar%20mitzvah"
                                        "&area=Brooklyn&guests=120&hours=4"}, headers=api_headers)

    @server.custom_route("/openapi.json", methods=["GET"], include_in_schema=False)
    async def openapi(request):  # noqa: ARG001
        paths: dict[str, Any] = {}
        components: dict[str, Any] = {}
        for n, t in (await tool_schemas()).items():
            params = []
            for key, prop in (t.input_schema.get("properties") or {}).items():
                prop = dict(prop)
                desc = prop.pop("description", None)
                prop.pop("title", None)
                types = [x for x in prop.pop("anyOf", []) if x.get("type") != "null"]
                if types:
                    prop.update(types[0])
                if prop.get("default", 0) is None:
                    prop.pop("default")
                param = {"name": key, "in": "query", "required": key in t.input_schema.get("required", []),
                         "schema": prop}
                if desc:
                    param["description"] = _fit(desc + (" Comma-separated." if prop.get("type") == "array" else ""), 300)
                if prop.get("type") == "array":
                    param.update(style="form", explode=False)
                params.append(param)
            out_schema = json.loads(json.dumps(t.output_schema or {"type": "object"}).replace(
                '"#/$defs/', '"#/components/schemas/'))
            components.update(out_schema.pop("$defs", {}))
            paths[f"/api/{n}"] = {"get": {
                "operationId": n, "summary": t.title or n, "description": _fit(t.description, 300), "parameters": params,
                "responses": {"200": {"description": "Result", "content": {"application/json": {"schema": out_schema}}},
                              "400": {"description": "Bad parameters; the error says what to pass."}}}}
        return JSONResponse({
            "openapi": "3.1.0",
            "info": {"title": "Agent Bookability Index API", "version": VERSION,
                     "description": server.instructions + "\nNo key or sign-up. The same tools are on the MCP server at "
                                    f"{BASE_URL}/mcp .", "contact": {"email": "info@therunup.ai"},
                     "license": {"name": "CC BY 4.0 (the compilation; quotes from vendors' own sites remain theirs)", "url": "https://creativecommons.org/licenses/by/4.0/"}},
            "servers": [{"url": BASE_URL}], "paths": paths, "components": {"schemas": components}},
            headers=doc_headers)

    @server.custom_route("/.well-known/mcp/server-card.json", methods=["GET"], include_in_schema=False)
    async def server_card(request):  # noqa: ARG001  MCP server card, for clients and crawlers that discover servers
        tools = await server.list_tools()
        return JSONResponse({
            "serverInfo": {"name": "ai.therunup.index/agent-bookability-index", "title": "Agent Bookability Index",
                           "version": VERSION},
            "description": server.instructions.split("\n")[0], "websiteUrl": f"{BASE_URL}/agents/",
            "iconUrl": f"{BASE_URL}/favicon.svg",
            "transport": {"type": "streamable-http", "url": f"{BASE_URL}/mcp"}, "authentication": {"required": False},
            "openapi": f"{BASE_URL}/openapi.json",
            "tools": [{"name": t.name, "title": t.title, "description": t.description} for t in tools]},
            headers=doc_headers)

    @server.custom_route("/.well-known/api-catalog", methods=["GET"], include_in_schema=False)
    async def api_catalog(request):  # noqa: ARG001  RFC 9727: where a site's APIs and their descriptions are
        return JSONResponse({"linkset": [
            {"anchor": f"{BASE_URL}/api",
             "service-desc": [{"href": f"{BASE_URL}/openapi.json", "type": "application/openapi+json"}],
             "service-doc": [{"href": f"{BASE_URL}/agents/", "type": "text/html"}]},
            {"anchor": f"{BASE_URL}/mcp",
             "service-desc": [{"href": f"{BASE_URL}/.well-known/mcp/server-card.json", "type": "application/json"}],
             "service-doc": [{"href": f"{BASE_URL}/agents/", "type": "text/html"}]}]},
            media_type="application/linkset+json", headers=doc_headers)

    @server.custom_route("/.well-known/mcp.json", methods=["GET"], include_in_schema=False)
    async def mcp_json(request):
        return await server_card(request)

    return server


class NoServerStream:
    """ASGI wrapper: answer GET /mcp with 405. The server is stateless and never sends anything on a server-to-client
    stream, so an open GET only held a connection for 60 seconds. The MCP spec lets a
    server without that stream return 405, and clients carry on without it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "GET" and scope["path"].rstrip("/") == "/mcp":
            resp = PlainTextResponse("No server-to-client stream here; send JSON-RPC requests with POST.\n",
                                     status_code=405, headers={"Allow": "POST"})
            await resp(scope, receive, send)
            return
        await self.app(scope, receive, send)


def http_app(server: MCPServer, host: str = "0.0.0.0"):
    """The app server.run("streamable-http", host=...) builds, wrapped. host decides DNS-rebinding protection the
    same way run() does: on for localhost, off for a public bind such as 0.0.0.0."""
    return NoServerStream(server.streamable_http_app(stateless_http=True, json_response=True, host=host))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP at /mcp instead of stdio")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    a = ap.parse_args()
    server = build_server()
    if a.http:  # stateless and plain JSON: any instance can answer any request, which suits any container host
        import uvicorn
        uvicorn.run(http_app(server, a.host), host=a.host, port=a.port, log_level="warning")
    else:
        server.run()


if __name__ == "__main__":
    main()
