# mcp-guard

A stateless security gateway for the Model Context Protocol (MCP), built on the **2026-07-28 spec**.
It sits between AI agents and MCP servers and decides, for every tool listing, tool call and tool
result, whether to **allow**, **ask a human** or **deny**. It uses deterministic policy first and a fast
decision model (TypeSafe's **Jev**, an LLM judge, a local classifier, or an offline rule baseline) for the
judgment calls, and it is evaluated against the **MCPTox** benchmark.

```
agent / MCP client ──► mcp-guard ──► MCP servers (GitHub, Postgres, internal tools…)
                        │  ├─ tools/list:  pin check (rug pulls) + description scan (tool poisoning)
                        │  ├─ tools/call:  policy → decision model → allow | ask (MRTR) | deny
                        │  ├─ results:     prompt-injection scan → pass | withhold
                        │  └─ every step:  JSONL audit + OpenTelemetry spans
```

## Why

- Researchers filed 30+ CVEs against MCP servers, clients and tooling in January–February 2026 alone.
- In MCPTox (AAAI 2026: 45 real MCP servers, 353 real tools, 1,312 attack cases), tool poisoning
  reached a 72.8% attack success rate, and the most cautious model refused fewer than 3% of attacks.
- The OWASP MCP Top 10 lists tool poisoning, prompt injection via tool output, rug pulls and
  over-permissioned tokens among the top risks.
- The 2026-07-28 spec made MCP stateless and put `Mcp-Method` / `Mcp-Name` in HTTP headers so
  gateways can route and authorize without parsing bodies. This project is that gateway.

## What it does

| Risk | Control |
|---|---|
| Tool poisoning (instructions hidden in descriptions) | Static scan (hidden Unicode, `<IMPORTANT>` tags, "don't tell the user") + decision-model questions; flagged tools are hidden from `tools/list` and their calls refused |
| Rug pulls (tool changes after review) | SHA-256 fingerprint of each tool; trust-on-first-use or a reviewed lockfile (`mcp-guard pin`); changed tools are quarantined |
| Dangerous calls | Hard rules (credential paths, secrets, `rm -rf`, `DROP TABLE`, `curl … \| sh`) deny before anything else |
| Exfiltration / privilege changes | External destinations outside an allowlist and permission changes escalate to a human |
| Calls the agent was tricked into | Typed yes/no risk questions to the decision model, optionally with the user's request as intent |
| Injection in tool output | Output scan; suspicious results are withheld or annotated before the model sees them |
| Header/body desync | `Mcp-Method` / `Mcp-Name` / version headers must match the JSON-RPC body (`-32020`) |
| Token passthrough | The client's `Authorization` header is never forwarded; upstream credentials come from config |
| Accountability | Append-only JSONL audit (argument hashes, not arguments) + OTel spans with `gen_ai.*` / `mcp.*` attributes |

### Human approval over MRTR

When a call needs a human, the gateway answers with an `InputRequiredResult` containing a form
elicitation and an HMAC-signed `requestState`. The client asks the user and retries with
`inputResponses`. Per the spec's MRTR security rules, the state binds the principal, server, tool,
a digest of the arguments and an expiry, and replays are rejected, so approving
`send_email(to=partner@…)` cannot be reused for `send_email(to=attacker@…)`. If the upstream server
uses MRTR itself, its `requestState` is carried inside the gateway's signed envelope, so both layers
survive the round trip. No session store is needed, so any replica can serve any retry.

## Quickstart

Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

mcp-guard demo-servers --port 9100           # terminal 1: an honest and a malicious MCP server
mcp-guard serve -c config/demo.yaml          # terminal 2: the gateway on :8800
mcp-guard demo                               # terminal 3: scripted walkthrough
```

Expected output:

```
1. list workspace tools       listed        visible: deploy, list_dir, read_file, run_shell, send_email, write_file
2. list weather tools         listed        visible: get_alerts, get_time | hidden: get_forecast (poisoned_description)
3. read notes                 allow         Q4 plan: ship the MCP gateway, …
4. read SSH key               blocked       credential or key path in arguments: /.ssh/, id_rsa
5. rm -rf /                   blocked       destructive command: rm -rf
6. email outside the org      allow (after 1 approval round)
7. read inbox (injected)      output withheld
8. call poisoned tool         blocked       tool is quarantined (poisoned_description)
9. weather alerts (injected)  output withheld   injection=0.90
10. deploy (server MRTR)      allow (after 1 approval round)   deployed build 1.4.2 to staging
11. rug pull, list again      listed        hidden: get_time (rug_pull), get_forecast (poisoned_description)
12. call rug-pulled tool      blocked       tool is quarantined (rug_pull)
```

With Docker (adds Jaeger for traces at http://localhost:16686):

```bash
docker compose up --build
docker compose run --rm demo
```

To use Jev: `MCPGUARD_BACKEND=jev TYPESAFE_API_KEY=… mcp-guard serve -c config/demo.yaml`.

## Decision backends

All backends answer the same typed questions, defined in [`guard/questions.py`](src/mcp_guard/guard/questions.py),
so they can be swapped and compared with everything else held constant.

| Backend | What it is | Notes |
|---|---|---|
| `heuristic` | Regex signals mapped to probabilities | Offline, free, the eval baseline and the runtime fallback |
| `jev` | TypeSafe Jev over `/v1/systemone` | Many yes/no questions answered in parallel in one call; works with any server that speaks the same format (e.g. open-weight decision models) via `base_url` |
| `llm` | Any OpenAI-compatible chat endpoint as a judge | Same questions, JSON probabilities |
| `classifier` | Local prompt-injection classifier (`pip install -e ".[classifier]"`) | Description/output gates; tool calls fall back to heuristic |

How Jev is used, following TypeSafe's own guidance for `jev-1.13`:

- Each question is one literal yes/no condition (Noul) with explicit `true` / `false` criteria.
  There are no compound or indirect questions, and no arithmetic or date logic, which stays in code.
- The state is trimmed to the fields a question needs, and secrets are redacted before it leaves
  the process.
- Jev's documentation says adversarial text can move its answers. So it never acts alone: hard
  policy runs first, thresholds map probabilities to allow/ask/deny, and the "ask" band keeps a human
  in the loop.
- If the model is unreachable, tool calls fail closed (`fail_mode: ask|deny`). Scans fail open with an
  audit entry, so an outage cannot hide every tool.

## Evaluation

```bash
git clone https://github.com/zhiqiangwang4/MCPTox-Benchmark ../MCPTox-Benchmark
mcp-guard mcptox --src ../MCPTox-Benchmark --out data/mcptox
mcp-guard eval -d data/mcptox/descriptions.jsonl -d data/mcptox/calls.jsonl --split test -b heuristic,jev,llm
mcp-guard eval -d data/mcptox/calls.jsonl --split test --intent -b heuristic,jev     # with user intent
mcp-guard eval ... --model-only                                                     # the model with no policy layer
```

The converter turns MCPTox into two gate-level datasets:

- **descriptions** (837): 485 poisoned descriptions vs. 352 genuine descriptions from the same servers.
- **calls** (4,175): tool calls that models actually made when a poisoning attack **succeeded**, vs. calls
  made when models ignored the poison and did the user's task. A call that appears on both sides is dropped.

Splits are by server (dev/test never share a server). Any rule tuning used only the dev split. The
MCPTox data is not redistributed; the converter runs on your clone.

**Baseline (heuristic backend, MCPTox test split, 2026-10-09).** Full reports are in [`reports/`](reports).

| Gate | Attacks | Attacks passed (ASR) | Escalated | Benign | Benign allowed | Benign escalated | Benign denied |
|---|---:|---:|---:|---:|---:|---:|---:|
| Descriptions (`tools/list`) | 212 | **12.7%** | n/a | 184 | **100%** | n/a | 0% |
| Calls, no intent | 839 | 62.9% | 32.8% | 937 | 86.3% | 13.7% | 0% |
| Calls, with user intent | 839 | **57.9%** | 37.8% | 937 | **91.9%** | 8.1% | 0% |

What this shows:

- Rules catch most **poisoned descriptions** (87% hidden, no false hides).
- Rules mostly stop calls that look dangerous: Code Injection 77%, Infrastructure Damage 71%,
  Credential Leakage 68%. They miss **semantic** attacks: Information Manipulation 21%, Data Tampering
  18%, Financial Loss 10%. Those calls look normal, such as a search query subtly changed or a different
  recipient. Catching them needs a model that compares the call to what the user asked for. That is the
  gap Jev or an LLM judge has to close, and the eval measures how much each closes per dollar and per
  millisecond.
- Passing the user's intent raises utility (fewer pointless approval prompts) and lowers ASR, even for
  the regex baseline.

"Escalated" counts as stopped pending a human. It is reported separately because a gate that escalates
everything would score 0% ASR and be useless.

The built-in `sample` set (`mcp-guard eval -d sample --split all`) also covers the **output** gate, which
MCPTox does not include. It is small and hand-written, for regression only.

## Configuration

See [`config/gateway.example.yaml`](config/gateway.example.yaml). All values accept `${ENV}` and `${ENV:-default}`.

```yaml
decision:
  backend: jev
  thresholds: {deny: 0.85, ask: 0.5}
policy:
  egress_allowlist: ["yourcompany.com"]
servers:
  github:
    url: https://api.githubcopilot.com/mcp/
    headers: {Authorization: "Bearer ${GITHUB_MCP_TOKEN}"}
    pinning: {mode: strict}        # only tools in the reviewed lockfile are exposed
    tools:
      delete_repository: {action: deny}
      merge_pull_request: {action: ask}
```

Clients reach a server at `POST /mcp/{server}`. An optional `_meta` key `dev.mcpguard/userIntent` carries the
user's request for intent-aware checks.

## Deploying

[`deploy/k8s`](deploy/k8s) has a Deployment (3+ replicas, non-root, read-only root filesystem), a Service,
an HPA, a PodDisruptionBudget and a ConfigMap. For several replicas:

- set the same `MCPGUARD_STATE_SECRET` everywhere, so approvals verify on any replica;
- use `pinning: strict` with a lockfile generated by `mcp-guard pin` and reviewed in a pull request, so
  every replica trusts the same tool definitions;
- send traces to your collector via `OTEL_EXPORTER_OTLP_ENDPOINT`.

## Limitations

- The gateway sees tool traffic, not the conversation. Without `userIntent` it cannot tell a legitimate
  but unusual call from an injected one.
- Replay protection is per replica (a shared nonce store such as Redis would make it global). The signed
  state still expires and is bound to the exact call.
- Trust-on-first-use pins are per replica and in memory; use strict mode in production.
- Output scanning withholds whole results; it does not surgically remove the injected span.
- The OpenTelemetry GenAI/MCP semantic conventions are still in Development status upstream; attribute
  names may change.
- Decision-model scores are vendor claims until you run the eval on your own traffic. Calibration holds
  statistically, not per answer.

## Project layout

```
src/mcp_guard/
  app.py            gateway (Starlette): routing, MRTR approvals, output scanning
  protocol.py       MCP 2026-07-28 wire helpers (headers, _meta, MRTR, errors)
  policy.py         deterministic rules and signal detectors
  pinning.py        tool fingerprints, TOFU/strict pinning, lockfile
  approvals.py      HMAC-signed requestState, replay guard, elicitation forms
  registry.py       vetted, cached view of each server's tools (uses ttlMs)
  guard/            questions, engine (thresholds, cache, fallback), backends: jev, llm_judge, classifier, heuristic
  eval/             dataset schema, MCPTox converter, runner and report
  demo/             honest + malicious demo MCP servers and a scripted client
tests/              92 tests: end-to-end through the gateway, policy, approvals, backends, evals
deploy/k8s/         Kubernetes manifests
```

## References

- MCP specification 2026-07-28: https://modelcontextprotocol.io/specification/2026-07-28 (changelog, MRTR pattern)
- MCPTox: A Benchmark for Tool Poisoning on Real-World MCP Servers (AAAI 2026): https://arxiv.org/abs/2508.14925
- OWASP MCP Top 10: https://owasp.org/www-project-mcp-top-10/
- TypeSafe Jev docs, API and `jev-1.13` limitations: https://docs.typesafe.ai
- OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
