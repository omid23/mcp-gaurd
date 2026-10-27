"""Command line: serve, pin, eval, mcptox, demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import Settings, load_settings


def _settings(path: str | None) -> Settings:
    return load_settings(path) if path else Settings()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .app import create_app

    settings = _settings(args.config)
    app = create_app(settings)
    uvicorn.run(app, host=args.host or settings.gateway.host, port=args.port or settings.gateway.port,
                log_level=args.log_level)


def cmd_pin(args: argparse.Namespace) -> None:
    from .pinning import write_lockfile
    from .registry import fetch_all_tools
    from .upstream import UpstreamClient

    settings = _settings(args.config)
    targets = args.server or sorted(settings.servers)

    async def run() -> dict[str, list[dict]]:
        upstream = UpstreamClient(settings.servers)
        try:
            return {s: (await fetch_all_tools(upstream, s))[0] for s in targets}
        finally:
            await upstream.aclose()

    tools = asyncio.run(run())
    write_lockfile(args.out, tools)
    for server, items in tools.items():
        print(f"pinned {len(items)} tools for '{server}'")
        for t in items:
            print(f"  - {t['name']}: {(t.get('description') or '')[:90]}")
    print(f"wrote {args.out}. Review the descriptions above before committing it.")


def cmd_eval(args: argparse.Namespace) -> None:
    from .eval.dataset import load_dataset
    from .eval.runner import run_eval, stratified_sample, to_markdown

    settings = _settings(args.config)
    cases = []
    for spec in args.dataset:
        cases.extend(load_dataset(spec))
    if args.split != "all":
        cases = [c for c in cases if c.split == args.split]
    if args.kind:
        cases = [c for c in cases if c.kind in args.kind]
    cases = stratified_sample(cases, args.limit)
    if not cases:
        sys.exit("no cases selected")
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    report = asyncio.run(run_eval(cases, backends, settings, use_intent=args.intent,
                                  model_only=args.model_only, concurrency=args.concurrency))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.name or "eval"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = to_markdown(report, title=args.title or f"Eval: {', '.join(args.dataset)} ({args.split})")
    (out / f"{stem}.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nwrote {out / (stem + '.json')} and {out / (stem + '.md')}")


def cmd_mcptox(args: argparse.Namespace) -> None:
    from .eval.dataset import write_jsonl
    from .eval.mcptox import convert, summarize

    descriptions, calls = convert(args.src)
    out = Path(args.out)
    write_jsonl(out / "descriptions.jsonl", descriptions)
    write_jsonl(out / "calls.jsonl", calls)
    print(f"descriptions: {len(descriptions)} {summarize(descriptions)}")
    print(f"calls:        {len(calls)} {summarize(calls)}")
    print(f"wrote {out / 'descriptions.jsonl'} and {out / 'calls.jsonl'}")


def cmd_demo_servers(args: argparse.Namespace) -> None:
    import uvicorn

    from .demo.servers import build_demo_app

    uvicorn.run(build_demo_app(), host=args.host, port=args.port, log_level="warning")


def cmd_demo(args: argparse.Namespace) -> None:
    from .demo.client import main as demo_main

    demo_main(args.gateway, args.upstream)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mcp-guard", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the gateway")
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("pin", help="fetch tool definitions and write a reviewable lockfile")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--server", action="append", help="server id (repeatable; default: all)")
    p.add_argument("--out", default="mcp-guard.lock")
    p.set_defaults(func=cmd_pin)

    p = sub.add_parser("eval", help="run eval cases through one or more decision backends")
    p.add_argument("--dataset", "-d", action="append", default=None,
                   help="'sample' or a JSONL path (repeatable)")
    p.add_argument("--backends", "-b", default="heuristic", help="comma list: heuristic,jev,llm,vertex,classifier")
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--kind", action="append", choices=["tool_call", "tool_description", "tool_output"])
    p.add_argument("--limit", type=int, default=0, help="stratified sample size (0 = all)")
    p.add_argument("--intent", action="store_true", help="give the guard the user's request as intent")
    p.add_argument("--model-only", action="store_true", help="disable deterministic policy and static checks")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="reports")
    p.add_argument("--name", default=None)
    p.add_argument("--title", default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("mcptox", help="convert a clone of MCPTox-Benchmark into eval JSONL")
    p.add_argument("--src", required=True, help="path to a clone of github.com/zhiqiangwang4/MCPTox-Benchmark")
    p.add_argument("--out", default="data/mcptox")
    p.set_defaults(func=cmd_mcptox)

    p = sub.add_parser("demo-servers", help="run the demo upstream MCP servers")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9100)
    p.set_defaults(func=cmd_demo_servers)

    p = sub.add_parser("demo", help="run the scripted demo against a running gateway")
    p.add_argument("--gateway", default="http://localhost:8800")
    p.add_argument("--upstream", default="http://localhost:9100")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    if args.cmd == "eval" and not args.dataset:
        args.dataset = ["sample"]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
