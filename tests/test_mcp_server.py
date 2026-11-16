"""The MCP server answers from recorded copies of the published JSON; no network in tests."""

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import server as srv

FIX = Path(__file__).parent / "fixtures" / "mcp"
FILES = {
    "/data/vendors.json": "vendors.json",
    "/prices/index.json": "prices.json",
    "/vendors/448-studio/index.json": "vendor-448-studio.json",
    "/cost/index.json": "cost.json",
}


def fake_fetch(path: str):
    if path not in FILES:
        req = httpx.Request("GET", f"https://index.therunup.ai{path}")
        raise httpx.HTTPStatusError("404", request=req, response=httpx.Response(404, request=req))
    return json.loads((FIX / FILES[path]).read_text())


@pytest.fixture
def call():
    server = srv.build_server(srv.IndexData(fetch=fake_fetch))

    def run(tool: str, **args):
        res = asyncio.run(server.call_tool(tool, args))
        out = json.loads(res.content[0].text)
        assert res.structured_content == out  # every tool returns typed, structured output
        return out

    return run


def test_tools_are_listed_read_only():
    server = srv.build_server(srv.IndexData(fetch=fake_fetch))
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"list_categories", "search_vendors", "shortlist_vendors", "get_vendor",
                                       "compare_vendors", "get_price_summary", "estimate_event_budget", "search",
                                       "fetch"}
    assert all(t.annotations.read_only_hint for t in tools)
    assert all(t.output_schema for t in tools)
    for t in tools:  # every parameter is described for the agent
        assert all(p.get("description") for p in t.input_schema.get("properties", {}).values()), t.name


def test_list_categories(call):
    out = call("list_categories")
    cats = {c["category"]: c for c in out["categories"]}
    assert cats["photographers"]["vendors"] == 4 and cats["photographers"]["with_price"] == 3
    assert cats["photographers"]["median_starting_price_usd"] == 3000
    assert "manhattan" in out["areas"]


def test_search_loose_words_area_and_price(call):
    out = call("search_vendors", category="wedding photographer", area="Brooklyn", max_price_usd=3000)
    names = [v["name"] for v in out["vendors"]]
    assert names == ["448 Studio"]  # in Brooklyn and at or under $3,000; unpriced vendors left out
    v = out["vendors"][0]
    assert v["starting_price"]["amount_usd"] == 2500 and v["starting_price"]["source"].startswith("https://")
    assert v["index_page"] == "https://index.therunup.ai/vendors/448-studio/"
    assert out["list_page"].endswith("/nyc/photographers/brooklyn/")


def test_search_sorted_by_score_then_price(call):
    out = call("search_vendors", category="photographers")
    assert [v["score"] for v in out["vendors"]] == sorted((v["score"] for v in out["vendors"]), reverse=True)
    # 448 Studio ($2,500) and AGD Festive Photography ($800) both score 38: the lower price comes first
    assert [v["name"] for v in out["vendors"][:2]] == ["AGD Festive Photography", "448 Studio"]
    assert out["vendors"][-1]["name"] == "4Eyes Photography"  # lowest score last


def test_search_books_online(call):
    out = call("search_vendors", category="planner", books_online=True)
    assert out["matched"] == 1 and out["vendors"][0]["how_to_book"]["online_booking"] is True


def test_search_unknown_category_and_area(call):
    with pytest.raises(ToolError, match="Use one of"):
        call("search_vendors", category="tattoo artists")
    with pytest.raises(ToolError, match="manhattan"):
        call("search_vendors", category="DJ", area="Chicago")


def test_search_limit_capped(call):
    out = call("search_vendors", category="florists", limit=500)
    assert out["shown"] == out["matched"] == 4


def test_get_vendor_by_name_url_and_slug(call):
    by_name = call("get_vendor", vendor="448 studio")
    by_url = call("get_vendor", vendor="https://index.therunup.ai/vendors/448-studio/")
    by_slug = call("get_vendor", vendor="448-studio")
    assert by_name == by_url == by_slug
    sp = by_name["facts"]["starting_price"]
    assert sp["quote"] and sp["source"] and sp["checked"]
    assert by_name["index_page"] == "https://index.therunup.ai/vendors/448-studio/"


def test_get_vendor_unknown(call):
    with pytest.raises(ToolError, match="No vendor"):
        call("get_vendor", vendor="No Such Vendor Anywhere")


def test_get_vendor_many_matches(call):
    with pytest.raises(ToolError, match="Pass one index_page"):
        call("get_vendor", vendor="photo")


def test_price_summary_sentence(call):
    out = call("get_price_summary", category="DJ")
    assert out["median"] == 2000 and out["vendors"] == 74
    assert out["sentence"].startswith("The median starting price DJs in the New York area publish")
    assert "November 2026" in out["sentence"] and out["source"].endswith("/prices/")


def test_price_summary_area(call):
    out = call("get_price_summary", category="photographers", area="Manhattan")
    assert out["area"] == "manhattan" and out["median"] == 3620 and "Manhattan" in out["sentence"]
    assert out["per_unit"] and out["per_unit"][0]["unit"] == "per hour"


def test_cache_reads_each_file_once():
    seen = []

    def counting(path):
        seen.append(path)
        return fake_fetch(path)

    data = srv.IndexData(fetch=counting)
    data.vendors(), data.vendors(), data.prices()
    assert seen == ["/data/vendors.json", "/prices/index.json"]


def test_each_call_logged(call, capsys):
    call("get_price_summary", category="DJ")
    call("search_vendors", category="florists", area="Manhattan")
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert [x["mcp_call"] for x in lines] == ["get_price_summary", "search_vendors"]
    assert {"category": "florists", "area": "Manhattan"}.items() <= lines[1]["args"].items()
    assert lines[1]["matched"] is not None


def test_search_all_categories(call):
    out = call("search_vendors", area="Manhattan", has_price=True, limit=25)
    assert out["category"] is None and len({v["category"] for v in out["vendors"]}) > 1
    assert out["list_page"] == "https://index.therunup.ai/"


def test_error_calls_are_logged(call, capsys):
    with pytest.raises(ToolError):
        call("search_vendors", category="tattoo artists")
    line = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")][-1]
    assert line["mcp_call"] == "search_vendors" and "Unknown category" in line["error"]


def test_shortlist_by_event_area_budget(call):
    out = call("shortlist_vendors", categories=["DJ", "photographer"], event_type="bar mitzvah", area="Brooklyn",
               budget_usd=3000, per_category=2)
    assert out["event_type"] == "bar-bat-mitzvahs" and out["area"] == "brooklyn"
    assert [g["category"] for g in out["groups"]] == ["djs", "photographers"]
    assert all(len(g["vendors"]) <= 2 for g in out["groups"])
    for g in out["groups"]:
        for v in g["vendors"]:
            assert "brooklyn" in v["areas"]


def test_compare_vendors(call):
    out = call("compare_vendors", vendors=["448-studio", "https://index.therunup.ai/vendors/448-studio/"])
    assert len(out["side_by_side"]) == 2 and out["side_by_side"][0]["name"] == "448 Studio"
    with pytest.raises(ToolError, match="between 2 and 5"):
        call("compare_vendors", vendors=["448-studio"])


def test_chatgpt_search_and_fetch(call):
    out = call("search", query="how much does a wedding photographer cost in Manhattan")
    ids = [r["id"] for r in out["results"]]
    assert ids[0] == "price:photographers:manhattan" and all(r["url"].startswith("https://") for r in out["results"])
    doc = call("fetch", id=ids[0])
    assert "Manhattan" in doc["text"] and doc["url"].endswith("/prices/")
    hits = call("search", query="448 studio")["results"]
    assert hits and hits[0]["id"] == "vendor:448-studio"
    doc = call("fetch", id="vendor:448-studio")
    assert doc["title"] == "448 Studio" and "starting price" in doc["text"] and doc["url"].endswith("/448-studio/")
    with pytest.raises(ToolError):
        call("fetch", id="nonsense")


def test_search_price_cap(call):
    hits = call("search", query="DJ under $2,000")["results"]
    assert hits and all(h["id"].startswith("vendor:") or h["id"].startswith("price:") for h in hits)


def test_prompt_and_resource():
    server = srv.build_server(srv.IndexData(fetch=fake_fetch))
    prompts = asyncio.run(server.list_prompts())
    assert [p.name for p in prompts] == ["plan_event_vendors"]
    res = asyncio.run(server.list_resources())
    assert [str(r.uri) for r in res] == ["index://about"]


def test_nyc_means_five_boroughs(call):
    out = call("search_vendors", category="DJ", area="NYC")
    assert out["area"] == "nyc" and out["list_page"].endswith("/nyc/djs/")
    assert all(set(v["areas"]) & {"manhattan", "brooklyn", "queens", "bronx", "staten-island"} for v in out["vendors"])
    assert call("get_price_summary", category="DJ", area="New York City")["area_note"].startswith("Prices are split")


def test_bronx_price_row_found(call):
    out = call("get_price_summary", category="photographers", area="the Bronx")
    assert out["area"] == "bronx" and "Bronx" in out["sentence"]


def test_search_does_not_match_generic_words_to_names(call):
    hits = call("search", query="best wedding photographer in Brooklyn")["results"]
    vend = [h for h in hits if h["id"].startswith("vendor:")]
    # ranked by bookability, not names containing "photo"/"wedding"; 448 Studio serves Brooklyn and has a price
    assert vend and "vendor:448-studio" in [h["id"] for h in vend]


def test_price_summary_points_to_cost_pages(call):
    out = call("get_price_summary", category="photographer")
    urls = [p["url"] for p in out["cost_pages"]]
    assert "https://index.therunup.ai/cost/event-photographer-per-hour/" in urls
    hits = call("search", query="how much does an event photographer cost per hour")["results"]
    cost = [h for h in hits if h["id"].startswith("cost:")]
    assert cost
    doc = call("fetch", id=cost[0]["id"])
    assert doc["url"].startswith("https://index.therunup.ai/cost/") and "median" in doc["text"]


def test_cost_page_for_the_area_asked_comes_first(call):
    out = call("get_price_summary", category="DJ", area="NYC")
    assert out["cost_pages"][0]["url"] == "https://index.therunup.ai/cost/wedding-dj/nyc/"


def test_shortlist_works_out_guests_and_hours(call):
    out = call("shortlist_vendors", categories=["photographer", "DJ"], area="Manhattan", guests=120, hours=4, per_category=5)
    assert out["guests"] == 120 and out["hours"] == 4
    for g in out["groups"]:
        for v in g["vendors"]:
            sp, est = v["starting_price"], v["estimate"]
            if sp and sp["per"] == "guest":
                assert est["total_usd"] == round(sp["amount_usd"] * 120, 2) and "120 guests" in est["basis"]
            if sp and sp["per"] == "hour":
                assert est["total_usd"] == round(sp["amount_usd"] * 4, 2)
            if sp and sp["per"] == "event":
                assert est["total_usd"] == sp["amount_usd"]


@pytest.fixture
def http():
    from starlette.testclient import TestClient
    server = srv.build_server(srv.IndexData(fetch=fake_fetch))
    return TestClient(srv.http_app(server))


def test_http_api_runs_tools_from_query_params(http):
    r = http.get("/api/shortlist_vendors?categories=DJ,photographer&area=Brooklyn&guests=120&hours=4&per_category=2")
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
    assert "no-store" in r.headers["cache-control"]  # never cached, so answers are current
    out = r.json()
    assert [g["category"] for g in out["groups"]] == ["djs", "photographers"] and out["guests"] == 120
    assert http.get("/api/get-vendor?vendor=448-studio").json()["name"] == "448 Studio"
    assert http.get("/api/search_vendors?category=DJ&books_online=true&max_price_usd=$5,000").status_code == 200
    bad = http.get("/api/search_vendors?category=zebra")
    assert bad.status_code == 400 and "Unknown category" in bad.json()["error"]
    assert http.get("/api/get_vendor").status_code == 400
    assert http.get("/api/nope").status_code == 404
    assert "shortlist_vendors" in http.get("/api").json()["tools"]
    assert http.get("/api/").status_code == 200 and http.get("/api/list_categories/").status_code == 200


def test_openapi_and_server_card(http):
    spec = http.get("/openapi.json").json()
    assert spec["openapi"].startswith("3.1") and "/api/shortlist_vendors" in spec["paths"]
    params = {p["name"]: p for p in spec["paths"]["/api/shortlist_vendors"]["get"]["parameters"]}
    assert params["categories"]["required"] and params["categories"]["schema"]["type"] == "array"
    assert "/api/search" not in spec["paths"] and "/api/fetch" not in spec["paths"]
    assert "$defs" not in json.dumps(spec)  # every reference resolves under components
    refs = set(re.findall(r'"#/components/schemas/([^"]+)"', json.dumps(spec)))
    assert refs and refs <= set(spec["components"]["schemas"])
    for path in spec["paths"].values():  # GPT Actions limits
        assert len(path["get"]["description"]) <= 300
        assert all(len(p.get("description", "")) <= 300 for p in path["get"]["parameters"])
    card = http.get("/.well-known/mcp/server-card.json").json()
    assert card["transport"]["url"].endswith("/mcp") and len(card["tools"]) == 9
    assert http.get("/.well-known/mcp.json").json() == card
    cat = http.get("/.well-known/api-catalog")
    assert cat.headers["content-type"].startswith("application/linkset+json")
    assert cat.json()["linkset"][0]["service-desc"][0]["href"].endswith("/openapi.json")


def test_estimate_units():
    def v(cat, per, amount=50.0):
        return {"category": cat, "price_from_usd": amount,
                "facts": {"starting_price": {"value": {"per": per}, "source": "https://x.example/"}}}
    assert srv.estimate(v("caterers", "person"), 80, None).total_usd == 4000
    assert srv.estimate(v("caterers", "guest"), 80, None).basis == "$50 per guest x 80 guests"
    assert srv.estimate(v("hair-and-makeup", "person"), 80, None) is None
    assert srv.estimate(v("photographers", "hour", 300), 80, 5).total_usd == 1500
    assert srv.estimate(v("djs", "event", 1500), None, None).total_usd == 1500
    assert srv.estimate(v("cakes-and-desserts", "item"), 80, None) is None


def test_estimate_event_budget(call):
    out = call("estimate_event_budget", categories=["DJ", "photographer", "florist"], hours=4)
    by = {x["category"]: x for x in out["services"]}
    assert by["photographers"]["median_usd"] == 1440 and "per hour x 4 hours" in by["photographers"]["basis"]
    assert by["djs"]["median_usd"] == 2000 and by["djs"]["basis"].startswith("published starting price")
    assert out["total_median_usd"] == 1440 + 2000 + 2500
    assert out["total_low_usd"] <= out["total_median_usd"] <= out["total_high_usd"]
    assert "$5,940" in out["sentence"] and "DJs, photographers and florists in the New York area" in out["sentence"]
    assert "in the Bronx" in call("estimate_event_budget", categories=["DJ"], area="bronx")["sentence"]
    whole = call("estimate_event_budget", categories=["photographer"])
    assert whole["services"][0]["median_usd"] == 3000
    with pytest.raises(ToolError, match="Unknown category"):
        call("estimate_event_budget", categories=["zebra"])


def test_needs_filter(call):
    all_djs = call("search_vendors", category="DJ", limit=25)["matched"]
    mc = call("search_vendors", category="DJ", needs=["MC"], limit=25)
    assert 0 < mc["matched"] < all_djs
    out = call("shortlist_vendors", categories=["DJ", "photographer"], needs=["MC", "second shooter"], per_category=5)
    by = {g["category"]: g for g in out["groups"]}
    assert by["djs"]["matched"] == mc["matched"]  # MC applies to DJs only
    assert 0 < by["photographers"]["matched"]
    with pytest.raises(ToolError, match="Unknown need"):
        call("search_vendors", needs=["unicorn"])


def test_area_words_cover_counties_and_towns():
    assert srv.area_key("Bergen County NJ") == "north-jersey"
    assert srv.area_key("Hoboken/Jersey City") == "north-jersey"
    assert srv.area_key("Essex County NJ (Montclair/Newark)") == "north-jersey"
    assert srv.area_key("Monmouth County NJ") == "central-jersey"
    assert srv.area_key("Nassau County") == "long-island"
    assert srv.area_key("White Plains") == "westchester"
    assert srv.area_key("Hudson Yards") != "north-jersey" and srv.area_key("Morris Park") != "north-jersey"
    assert srv.area_key("Ocean Parkway") != "central-jersey"


def test_text_needs_lighting():
    lit = {"category": "djs", "facts": {"equipment_included": {"value": "Sound, 4 multi-colored up lights"}}}
    dark = {"category": "djs", "facts": {"equipment_included": {"value": "Two speakers and a microphone"}}}
    unknown = {"category": "djs", "facts": {}}
    keys = srv.need_keys(["uplighting"])
    assert keys == ["lighting"]
    assert srv.has_needs(lit, keys) and not srv.has_needs(dark, keys) and not srv.has_needs(unknown, keys)
    assert srv.need_keys(["Photo Booth"]) == ["photo_booth"]


def test_card_price_from_other_price_fields():
    v = {"category": "caterers", "price_from_usd": 50.0, "facts": {
        "minimum_spend": {"value": {"amount": 500, "per": "event"}},
        "price_per_guest": {"value": {"amount": 50, "per": "person", "note": "per adult"}, "source": "https://x.example/b"}}}
    p = srv._price(v)
    assert p.amount_usd == 50 and p.per == "person" and p.covers == "per adult" and p.source == "https://x.example/b"
    assert srv.estimate(v, 150, None).total_usd == 7500
    rng = {"category": "djs", "price_from_usd": 1200.0, "facts": {"typical_price_range": {"value": {"min": 1200, "max": 3000}}}}
    assert srv._price(rng).per == "event"
    pk = {"category": "djs", "price_from_usd": 900.0, "facts": {"packages": {"value": [{"name": "A", "price": 900}]}}}
    assert srv._price(pk).covers == "lowest listed package"
    assert srv._price({"category": "djs", "price_from_usd": None, "facts": {}}) is None


def test_new_jersey_area(call):
    assert srv.area_key("New Jersey") == "new-jersey" and srv.area_key("NJ") == "new-jersey"
    assert srv.area_key("Bergen County NJ") == "north-jersey"  # the county wins over the state
    nj = {"areas": ["central-jersey"]}
    assert srv.in_area(nj, "new-jersey") and not srv.in_area({"areas": ["queens"]}, "new-jersey")
    out = call("search_vendors", category="DJ", area="New Jersey")
    assert out["area"] == "new-jersey"
    with pytest.raises(ToolError, match="New York area only"):
        call("search_vendors", area="Austin")


def test_officiants_are_not_listed_as_planners():
    assert not srv.listed({"category": "planners", "facts": {"planner_type": {"value": "officiant"}}})
    assert srv.listed({"category": "planners", "facts": {"planner_type": {"value": '"full-service planner"'}}})
    assert srv.listed({"category": "planners", "facts": {}})
    assert srv.listed({"category": "djs", "facts": {"planner_type": {"value": "officiant"}}})


def test_get_mcp_is_405_and_post_still_works():
    from starlette.testclient import TestClient
    app = srv.http_app(srv.build_server(srv.IndexData(fetch=fake_fetch)))
    with TestClient(app) as http:  # runs the app's lifespan, as uvicorn does
        r = http.get("/mcp", headers={"Accept": "text/event-stream"})
        assert r.status_code == 405 and r.headers["allow"] == "POST"
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18",
                "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
        r = http.post("/mcp", json=init, headers={"Accept": "application/json, text/event-stream"})
        assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "agent-bookability-index"


def test_search_pages_and_coverage(call):
    first = call("search_vendors", category="DJ", limit=2)
    assert first["offset"] == 0 and first["shown"] == 2
    if first["matched"] > 2:
        nxt = call("search_vendors", category="DJ", limit=2, offset=first["next_offset"])
        assert nxt["offset"] == 2 and {v["name"] for v in nxt["vendors"]}.isdisjoint({v["name"] for v in first["vendors"]})
    last = call("search_vendors", category="DJ", limit=25, offset=first["matched"] - 1)
    assert last["next_offset"] is None and last["shown"] == 1
    cov = call("list_categories")
    assert "new-jersey" in cov["areas"] and "kosher" in cov["needs"] and cov["event_types"]


def test_search_sort_by_price(call):
    out = call("search_vendors", category="DJ", sort="price", has_price=True, limit=25)
    prices = [v["starting_price"]["amount_usd"] for v in out["vendors"]]
    assert prices == sorted(prices)
    with pytest.raises(ToolError, match="sort must be"):
        call("search_vendors", sort="cheapest")


def test_venues_get_a_clear_answer(call):
    with pytest.raises(ToolError, match="not venues"):
        call("search_vendors", category="wedding venue")


def test_free_text_search_reads_needs(call):
    out = call("search", query="DJ with an MC on Long Island")
    names = {r["title"].split(" (")[0] for r in out["results"] if r["id"].startswith("vendor:")}
    assert names and names <= {"A Night To Remember Entertainment", "Absolute Entertainment"}  # the DJs that MC
