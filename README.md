# Agent Bookability Index MCP server

A read-only MCP server that lets AI agents search, compare and price New York area event vendors (photographers, DJs, caterers, florists, planners, videographers, live bands, photo booths, hair and makeup artists, and cake and dessert makers). Every answer comes from the vendor's own published prices and pages, with the source page to cite and the date it was read.

It reads only what [index.therunup.ai](https://index.therunup.ai/) already publishes, so it needs no key, no database and no account.

## Use the hosted server

No install needed:

```
https://index.therunup.ai/mcp
```

Streamable HTTP, no authentication. Setup for Claude, ChatGPT, Cursor, VS Code and Gemini CLI: https://index.therunup.ai/agents/ . Listed in the official MCP Registry as `ai.therunup.index/agent-bookability-index`.

## Tools

- `shortlist_vendors`: the most bookable matching vendors for each service an event needs, by area, event type, budget, guests and hours, with a cost estimate per vendor.
- `search_vendors`: vendors in one category, filtered by area, price, online booking, needs (kosher, halal, MC, second shooter, lighting) or name.
- `get_vendor`: every fact the index holds for one vendor, each with the quote from its site, the source page and the date read.
- `compare_vendors`: two to five vendors side by side.
- `get_price_summary`: what a service costs in an area (median and middle half of published prices), in one sentence with its source.
- `estimate_event_budget`: what all of an event's services cost together.
- `list_categories`: categories with vendor counts and median starting prices.
- `search` and `fetch`: the shapes ChatGPT expects.

Category and area words are loose: "wedding photographer", "DJ", "live band"; "Brooklyn", "the Hamptons", "Jersey City".

## Run it yourself

```bash
uv run python -m mcp_server.server            # stdio, for Claude Desktop or Claude Code
uv run python -m mcp_server.server --http     # streamable HTTP at http://0.0.0.0:8080/mcp (PORT is honoured)
uv run pytest -q                              # tests, from recorded copies of the published JSON
```

`INDEX_BASE_URL` points it at another copy of the site.

## The data

Prices by service and area: [NYC Event Price Index](https://index.therunup.ai/prices/), with a page per service such as [what a caterer costs per guest](https://index.therunup.ai/cost/caterer-per-guest/) or [what a wedding DJ costs in NYC](https://index.therunup.ai/cost/wedding-dj/nyc/). The open data (CC BY 4.0): [GitHub](https://github.com/therunup/nyc-event-vendor-index), [Hugging Face](https://huggingface.co/datasets/therunup/nyc-event-vendor-index), [Zenodo](https://doi.org/10.5281/zenodo.24274058). Method: https://index.therunup.ai/method/ .

## Logging, licence and corrections

For each tool call, the hosted server's application log records: the tool, its arguments (including any search text), the client's name and version, its user agent, any error, how many results matched and the vendor looked up. The application does not log IP addresses; the hosting provider's own request logs may record them.

The code is MIT licensed. The data it serves is our compilation, licensed CC BY 4.0; quotes from vendors' own websites, and vendors' names and trademarks, remain theirs. The test fixtures under `tests/fixtures/` are an older snapshot of that published data, kept only for tests (two vendors renamed to fictional examples), and are not MIT. Provided as is, without warranty; no vendor listed is affiliated with or endorses The Runup.

The score measures how much an AI agent can learn and do on a vendor's own site, not the quality of the vendor's work. Only some vendors publish prices, so price figures describe those vendors.

The Runup, which runs this index, also builds software for event vendors. The same rules and scoring apply to every vendor, and no vendor can pay for a higher score or a better place in a list.

This repository mirrors the server that runs at index.therunup.ai/mcp. Any vendor, including a sole proprietor listed under their own name, can correct or remove their listing at https://index.therunup.ai/claim/ or by emailing info@therunup.ai.
