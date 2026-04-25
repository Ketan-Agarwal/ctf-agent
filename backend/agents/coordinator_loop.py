"""Shared coordinator event loop — used by both Claude SDK and Codex coordinators."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt

from backend.config import Settings
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.deps import CoordinatorDeps
from backend.hitl import HITLGate, _suppress_stream_handlers, _restore_stream_handlers
from backend.models import DEFAULT_MODELS
from backend.poller import CTFdPoller
from backend.prompts import ChallengeMeta

_console = Console()

logger = logging.getLogger(__name__)

# Callable type for a coordinator turn: (message) -> None
TurnFn = Callable[[str], Coroutine[Any, Any, None]]


def build_deps(
    settings: Settings,
    model_specs: list[str] | None = None,
    challenges_root: str = "challenges",
    no_submit: bool = False,
    challenge_dirs: dict[str, str] | None = None,
    challenge_metas: dict[str, ChallengeMeta] | None = None,
) -> tuple[CTFdClient, CostTracker, CoordinatorDeps]:
    """Create CTFd client, cost tracker, and coordinator deps."""
    ctfd = CTFdClient(
        base_url=settings.ctfd_url,
        token=settings.ctfd_token,
        username=settings.ctfd_user,
        password=settings.ctfd_pass,
    )
    cost_tracker = CostTracker()
    specs = model_specs or list(DEFAULT_MODELS)
    Path(challenges_root).mkdir(parents=True, exist_ok=True)

    deps = CoordinatorDeps(
        ctfd=ctfd,
        cost_tracker=cost_tracker,
        settings=settings,
        model_specs=specs,
        challenges_root=challenges_root,
        no_submit=no_submit,
        max_concurrent_challenges=getattr(settings, "max_concurrent_challenges", 10),
        challenge_dirs=challenge_dirs or {},
        challenge_metas=challenge_metas or {},
        hitl_gate=HITLGate(enabled=getattr(settings, "hitl", False)),
    )

    # Pre-load already-pulled challenges
    for d in Path(challenges_root).iterdir():
        meta_path = d / "metadata.yml"
        if meta_path.exists():
            meta = ChallengeMeta.from_yaml(meta_path)
            if meta.name not in deps.challenge_dirs:
                deps.challenge_dirs[meta.name] = str(d)
                deps.challenge_metas[meta.name] = meta

    return ctfd, cost_tracker, deps


async def run_event_loop(
    deps: CoordinatorDeps,
    ctfd: CTFdClient,
    cost_tracker: CostTracker,
    turn_fn: TurnFn,
    status_interval: int = 60,
) -> dict[str, Any]:
    """Run the shared coordinator event loop.

    Args:
        deps: Coordinator dependencies (shared state).
        ctfd: CTFd client (for poller).
        cost_tracker: Cost tracker.
        turn_fn: Async function that sends a message to the coordinator LLM.
        status_interval: Seconds between status updates.
    """
    poller = CTFdPoller(ctfd=ctfd, interval_s=5.0)
    await poller.start()

    # Start operator message HTTP endpoint
    msg_server = await _start_msg_server(deps.operator_inbox, deps, deps.msg_port)

    logger.info(
        "Coordinator starting: %d models, %d challenges, %d solved",
        len(deps.model_specs),
        len(poller.known_challenges),
        len(poller.known_solved),
    )

    unsolved = poller.known_challenges - poller.known_solved
    initial_msg = (
        f"CTF is LIVE. {len(poller.known_challenges)} challenges, "
        f"{len(poller.known_solved)} solved.\n"
        f"Unsolved: {sorted(unsolved) if unsolved else 'NONE'}\n"
        "Fetch challenges and spawn swarms for all unsolved."
    )

    # Ctrl+C injection state
    inject_requested = False
    last_sigint_time = 0.0

    def _sigint_handler(sig, frame):
        nonlocal inject_requested, last_sigint_time
        now = time.monotonic()
        if now - last_sigint_time < 2.0:
            # Double Ctrl+C within 2s → real shutdown
            raise KeyboardInterrupt
        last_sigint_time = now
        inject_requested = True

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        await turn_fn(initial_msg)

        # Auto-spawn swarms for unsolved challenges if coordinator LLM didn't
        await _auto_spawn_unsolved(deps, poller)

        last_status = asyncio.get_event_loop().time()

        while True:
            # ---- Ctrl+C inject prompt ----
            if inject_requested:
                inject_requested = False
                await _interactive_inject(deps)

            events = []
            evt = await poller.get_event(timeout=5.0)
            if evt:
                events.append(evt)
            events.extend(poller.drain_events())

            # Auto-kill swarms for solved challenges
            for evt in events:
                if evt.kind == "challenge_solved" and evt.challenge_name in deps.swarms:
                    swarm = deps.swarms[evt.challenge_name]
                    if not swarm.cancel_event.is_set():
                        swarm.kill()
                        logger.info("Auto-killed swarm for: %s", evt.challenge_name)

            parts: list[str] = []
            for evt in events:
                if evt.kind == "new_challenge":
                    parts.append(f"NEW CHALLENGE: '{evt.challenge_name}' appeared. Spawn a swarm.")
                    # Auto-spawn for new challenges
                    await _auto_spawn_one(deps, evt.challenge_name)
                elif evt.kind == "challenge_solved":
                    parts.append(f"SOLVED: '{evt.challenge_name}' — swarm auto-killed.")

            # Detect finished swarms
            for name, task in list(deps.swarm_tasks.items()):
                if task.done():
                    parts.append(f"SOLVER FINISHED: Swarm for '{name}' completed. Check results or retry.")
                    deps.swarm_tasks.pop(name, None)

            # Drain solver-to-coordinator messages
            while True:
                try:
                    solver_msg = deps.coordinator_inbox.get_nowait()
                    parts.append(f"SOLVER MESSAGE: {solver_msg}")
                except asyncio.QueueEmpty:
                    break

            # Drain operator messages
            while True:
                try:
                    op_msg = deps.operator_inbox.get_nowait()
                    parts.append(f"OPERATOR MESSAGE: {op_msg}")
                    logger.info("Operator message: %s", op_msg[:200])
                except asyncio.QueueEmpty:
                    break

            # Periodic status update — only when there are active swarms or other events
            now = asyncio.get_event_loop().time()
            if now - last_status >= status_interval:
                last_status = now
                active = [n for n, t in deps.swarm_tasks.items() if not t.done()]
                solved_set = poller.known_solved
                unsolved_set = poller.known_challenges - solved_set
                status_line = (
                    f"STATUS: {len(solved_set)} solved, {len(unsolved_set)} unsolved, "
                    f"{len(active)} active swarms. Cost: ${cost_tracker.total_cost_usd:.2f}"
                )
                # Only send to coordinator if there's something happening
                if active or parts:
                    parts.append(status_line)
                else:
                    logger.info(f"Event -> coordinator: {status_line}")

            if parts:
                msg = "\n\n".join(parts)
                logger.info("Event -> coordinator: %s", msg[:200])
                await turn_fn(msg)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Coordinator shutting down...")
    except _InjectShutdown:
        logger.info("Coordinator shutting down (from inject prompt)...")
    except Exception as e:
        logger.error("Coordinator fatal: %s", e, exc_info=True)
    finally:
        if msg_server:
            msg_server.close()
            await msg_server.wait_closed()
        await poller.stop()
        for swarm in deps.swarms.values():
            swarm.kill()
        for task in deps.swarm_tasks.values():
            task.cancel()
        if deps.swarm_tasks:
            await asyncio.gather(*deps.swarm_tasks.values(), return_exceptions=True)
        cost_tracker.log_summary()
        signal.signal(signal.SIGINT, original_handler)
        try:
            await ctfd.close()
        except Exception:
            pass

    return {
        "results": deps.results,
        "total_cost_usd": cost_tracker.total_cost_usd,
        "total_tokens": cost_tracker.total_tokens,
    }


class _InjectShutdown(Exception):
    """Raised from inject prompt when user wants to quit."""


async def _inject_into_challenge(deps: CoordinatorDeps, challenge_name: str, hint: str) -> str:
    """Broadcast an operator hint into a challenge's message bus."""
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No active swarm for '{challenge_name}'"
    await swarm.message_bus.broadcast(
        f"**🎯 OPERATOR HINT**: {hint}",
        source="operator",
    )
    logger.info("Operator hint injected into %s: %s", challenge_name, hint[:200])
    return f"Hint injected into {challenge_name}"


def _interactive_inject_blocking(challenge_names: list[str]) -> tuple[str, str] | None:
    """Blocking Rich prompt for inject — runs via asyncio.to_thread.

    Returns (challenge_name, hint) or None if cancelled.
    """
    if not challenge_names:
        _console.print("[red]No active swarms to inject into.[/red]")
        return None

    # Build numbered list
    lines = []
    for i, name in enumerate(challenge_names, 1):
        lines.append(f"  [bold][{i}][/bold] {name}")
    panel = Panel(
        "\n".join(lines),
        title="[magenta]💉 Inject Hint (Ctrl+C again to quit)[/magenta]",
        border_style="magenta",
    )
    _console.print()
    _console.print(panel)

    try:
        if len(challenge_names) == 1:
            choice = 1
            _console.print(f"  Auto-selected: [bold]{challenge_names[0]}[/bold]")
        else:
            choice = IntPrompt.ask(
                "[magenta]Select challenge[/magenta]",
                choices=[str(i) for i in range(1, len(challenge_names) + 1)],
            )
        challenge = challenge_names[choice - 1]
        hint = Prompt.ask(f"[magenta]Hint for {challenge}[/magenta]")
        if not hint.strip():
            _console.print("[yellow]Empty hint — cancelled.[/yellow]")
            return None
        return challenge, hint
    except (KeyboardInterrupt, EOFError):
        return None


async def _interactive_inject(deps: CoordinatorDeps) -> None:
    """Show interactive inject prompt (triggered by Ctrl+C)."""
    active = [
        name for name, swarm in deps.swarms.items()
        if not swarm.cancel_event.is_set()
    ]
    saved = _suppress_stream_handlers()
    try:
        result = await asyncio.to_thread(_interactive_inject_blocking, active)
    except KeyboardInterrupt:
        _restore_stream_handlers(saved)
        raise _InjectShutdown()
    finally:
        _restore_stream_handlers(saved)

    if result:
        challenge_name, hint = result
        msg = await _inject_into_challenge(deps, challenge_name, hint)
        _console.print(f"[green]✅ {msg}[/green]\n")


async def _auto_spawn_one(deps: CoordinatorDeps, challenge_name: str) -> None:
    """Auto-spawn a swarm for a single challenge if not already running."""
    if challenge_name in deps.swarms:
        return
    if challenge_name in deps.denied_spawns:
        return
    active = sum(1 for t in deps.swarm_tasks.values() if not t.done())
    if active >= deps.max_concurrent_challenges:
        return
    # Global cost cap — don't spawn new challenges if budget is exhausted
    cost_limit = getattr(deps.settings, "cost_limit_global", 50.0)
    if deps.cost_tracker.total_cost_usd >= cost_limit:
        logger.warning(
            f"Global cost limit ${cost_limit:.2f} reached (${deps.cost_tracker.total_cost_usd:.2f} spent) — not spawning {challenge_name}"
        )
        return
    try:
        from backend.agents.coordinator_core import do_spawn_swarm
        result = await do_spawn_swarm(deps, challenge_name)
        logger.info(f"Auto-spawn {challenge_name}: {result[:100]}")
    except Exception as e:
        logger.warning(f"Auto-spawn failed for {challenge_name}: {e}")


async def _auto_spawn_unsolved(deps: CoordinatorDeps, poller) -> None:
    """Auto-spawn swarms for all unsolved challenges that don't have active swarms."""
    unsolved = poller.known_challenges - poller.known_solved
    for name in sorted(unsolved):
        await _auto_spawn_one(deps, name)

async def _start_msg_server(
    inbox: asyncio.Queue,
    deps: CoordinatorDeps,
    port: int = 0,
) -> asyncio.Server | None:
    """Start a tiny HTTP server that accepts operator messages and hint injections.

    POST /msg     {"message": "..."}                     → operator inbox (coordinator)
    POST /inject  {"challenge": "...", "message": "..."}  → solver message bus (direct)
    """

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            # Read HTTP request
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            headers: dict[str, str] = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break
                if b":" in line:
                    k, v = line.decode().split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            decoded = request_line.decode() if request_line else ""
            parts = decoded.split()
            method = parts[0] if parts else ""
            path = parts[1] if len(parts) > 1 else "/"
            content_length = int(headers.get("content-length", 0))

            if method != "POST" or content_length <= 0:
                resp = json.dumps({"error": "POST with JSON body required"})
                writer.write(f"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}".encode())
                await writer.drain()
                return

            body = await asyncio.wait_for(reader.read(content_length), timeout=5)
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {"message": body.decode("utf-8", errors="replace")}

            if path.rstrip("/") == "/inject":
                # Direct solver injection
                challenge = data.get("challenge", "")
                message = data.get("message", "")
                if not challenge or not message:
                    resp = json.dumps({"error": "Both 'challenge' and 'message' required"})
                    writer.write(f"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}".encode())
                else:
                    result = await _inject_into_challenge(deps, challenge, message)
                    resp = json.dumps({"ok": True, "result": result})
                    writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}".encode())
            else:
                # Default: operator message to coordinator
                message = data.get("message", json.dumps(data))
                inbox.put_nowait(message)
                resp = json.dumps({"ok": True, "queued": message[:200]})
                writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}".encode())

            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    try:
        server = await asyncio.start_server(_handle, "127.0.0.1", port)
        actual_port = server.sockets[0].getsockname()[1]
        logger.info(f"Operator message endpoint listening on http://127.0.0.1:{actual_port}")
        return server
    except OSError as e:
        logger.warning(f"Could not start operator message endpoint: {e}")
        return None
