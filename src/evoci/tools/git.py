"""Read-only Git helpers layered over the constrained command runner."""

from __future__ import annotations

from evoci.tools.shell import CommandResult, CommandRunner


class GitTools:
    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    async def status(self) -> CommandResult:
        return await self.runner.run(["git", "status", "--short", "--branch"])

    async def log(self, limit: int = 20) -> CommandResult:
        return await self.runner.run(
            ["git", "log", f"-{min(max(limit, 1), 100)}", "--oneline", "--decorate"]
        )

    async def show(self, revision: str = "HEAD") -> CommandResult:
        if revision.startswith("-") or any(char.isspace() for char in revision):
            raise ValueError("invalid revision")
        return await self.runner.run(["git", "show", "--stat", "--oneline", revision])

    async def diff(self) -> CommandResult:
        return await self.runner.run(["git", "diff", "--no-ext-diff", "--"])
