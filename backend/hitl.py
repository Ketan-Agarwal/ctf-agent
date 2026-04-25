"""Human-in-the-Loop approval gates for CTF agent decisions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

if TYPE_CHECKING:
    from backend.cost_tracker import CostTracker

logger = logging.getLogger(__name__)
_console = Console()


def _suppress_stream_handlers() -> list[tuple[logging.Handler, int]]:
    """Temporarily raise all StreamHandler levels to CRITICAL so log output
    doesn't interleave with interactive HITL prompts.  Returns the original
    levels so they can be restored."""
    saved: list[tuple[logging.Handler, int]] = []
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            saved.append((handler, handler.level))
            handler.setLevel(logging.CRITICAL + 1)
    return saved


def _restore_stream_handlers(saved: list[tuple[logging.Handler, int]]) -> None:
    """Restore handler levels saved by ``_suppress_stream_handlers``."""
    for handler, level in saved:
        handler.setLevel(level)


def _ask(prompt: str) -> bool:
    """Blocking Rich confirm prompt (run via asyncio.to_thread)."""
    return Confirm.ask(prompt, default=False)


class HITLGate:
    """Approval gates that block on human input when HITL is enabled.

    When ``enabled=False`` every gate auto-approves immediately.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _prompt(self, message: str) -> bool:
        if not self.enabled:
            return True
        _console.print()
        saved = _suppress_stream_handlers()
        try:
            return await asyncio.to_thread(_ask, message)
        finally:
            _restore_stream_handlers(saved)

    # ------------------------------------------------------------------
    # Public gates
    # ------------------------------------------------------------------

    async def approve_spawn(
        self,
        challenge_name: str,
        category: str,
        points: int,
        n_models: int,
    ) -> bool:
        """Ask before spawning a swarm for a challenge."""
        panel = Panel(
            f"[bold]{challenge_name}[/bold]  ({category}, {points} pts)\n"
            f"Models: {n_models}",
            title="[yellow]⚡ Spawn Swarm?[/yellow]",
            border_style="yellow",
        )
        _console.print(panel)
        return await self._prompt("[yellow]Approve spawn?[/yellow]")

    async def approve_flag_submit(
        self,
        challenge_name: str,
        flag: str,
        model_spec: str,
        cost_so_far: float,
    ) -> bool:
        """Ask before submitting a flag to CTFd."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("Challenge", f"[bold]{challenge_name}[/bold]")
        table.add_row("Flag", f"[green]{flag}[/green]")
        table.add_row("Found by", model_spec)
        table.add_row("Cost so far", f"${cost_so_far:.2f}")
        panel = Panel(table, title="[cyan]🏁 Submit Flag?[/cyan]", border_style="cyan")
        _console.print(panel)
        return await self._prompt("[cyan]Submit this flag?[/cyan]")

    async def approve_bump(
        self,
        challenge_name: str,
        model_spec: str,
        bump_count: int,
        cost_so_far: float,
        findings: str,
    ) -> bool:
        """Ask before bumping a solver that has already been bumped N times."""
        panel = Panel(
            f"[bold]{challenge_name}[/bold] / {model_spec}\n"
            f"Bumps so far: {bump_count}  |  Cost: ${cost_so_far:.2f}\n"
            f"Findings: {findings[:200] or 'none'}",
            title="[magenta]🔄 Bump Solver?[/magenta]",
            border_style="magenta",
        )
        _console.print(panel)
        return await self._prompt("[magenta]Allow another bump?[/magenta]")

    async def approve_continue(
        self,
        challenge_name: str,
        cost: float,
        limit: float,
    ) -> bool:
        """Ask when a challenge exceeds its cost limit."""
        panel = Panel(
            f"[bold]{challenge_name}[/bold]\n"
            f"Spent: [red]${cost:.2f}[/red]  |  Limit: ${limit:.2f}",
            title="[red]💰 Cost Limit Exceeded[/red]",
            border_style="red",
        )
        _console.print(panel)
        return await self._prompt("[red]Continue spending on this challenge?[/red]")
