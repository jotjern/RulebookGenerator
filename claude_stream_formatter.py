import json

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class ClaudeStreamFormatter:
    """Rich formatter for Claude Code stream-json events."""

    def __init__(self):
        self.console = Console()
        self.tool_input_buffers = {}
        self.last_full_text = ""

    def print_start(self, command: list[str], cwd):
        command_text = " ".join(command[:1] + ["..."] + command[-2:])
        table = Table.grid(expand=False)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("cwd", str(cwd))
        table.add_row("cmd", command_text)
        self.console.print(
            Panel(table, title="[bold cyan]Claude Code[/bold cyan]", border_style="cyan")
        )

    def handle_raw_line(self, raw_line: str):
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            self._print_line("raw", raw_line, "yellow")
            return

        event_type = event.get("type", "event")
        if event_type == "stream_event":
            self._handle_stream_event(event)
        elif event_type == "system":
            self._print_line(
                event.get("subtype", "system"),
                f"session={event.get('session_id', '')}",
                "cyan",
            )
        elif event_type == "assistant":
            self._handle_assistant(event.get("message", {}))
        elif event_type == "result":
            self._handle_result(event)
        elif event_type == "rate_limit_event":
            self._handle_rate_limit(event)
        elif event_type in {
            "user",
            "message_start",
            "message_delta",
            "message_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "ping",
        }:
            return
        else:
            self._print_json(event_type, event)

    def _handle_stream_event(self, event: dict):
        stream_event = event.get("event", {})
        stream_type = stream_event.get("type")

        if stream_type == "content_block_start":
            block = stream_event.get("content_block", {})
            if block.get("type") == "tool_use":
                self.tool_input_buffers[stream_event.get("index")] = {
                    "name": block.get("name", "tool"),
                    "json": "",
                }
            return

        if stream_type == "content_block_delta":
            index = stream_event.get("index")
            delta = stream_event.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "").strip()
                if text:
                    self._print_line("text", text, "white")
            elif delta.get("type") == "input_json_delta" and index in self.tool_input_buffers:
                self.tool_input_buffers[index]["json"] += delta.get("partial_json", "")
            return

        if stream_type == "content_block_stop":
            buffered = self.tool_input_buffers.pop(stream_event.get("index"), None)
            if buffered:
                tool_input = self._parse_tool_json(buffered["json"])
                self._print_tool(buffered["name"], tool_input)
            return

        if stream_type in {
            "message_start",
            "message_delta",
            "message_stop",
            "content_block_start",
            "ping",
        }:
            return

        self._print_line("stream", stream_type or "unknown", "dim")

    def _handle_assistant(self, message: dict):
        for block in message.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "").strip()
                if text and text != self.last_full_text:
                    self.last_full_text = text
                    self._print_line("text", text, "white")
            elif block_type == "tool_use":
                self.last_full_text = ""
                self._print_tool(block.get("name", "tool"), block.get("input", {}))

    def _handle_result(self, event: dict):
        bits = []
        if isinstance(event.get("total_cost_usd"), (int, float)):
            bits.append(f"${event['total_cost_usd']:.4f}")
        if isinstance(event.get("duration_ms"), int):
            bits.append(f"{event['duration_ms'] / 1000:.1f}s")
        self._print_line(event.get("subtype", "result"), "  ".join(bits), "green")
        result = event.get("result", "").strip()
        if result:
            self._print_line("result", result, "green")

    def _handle_rate_limit(self, event: dict):
        info = event.get("rate_limit_info", {})
        resets_at = info.get("resetsAt")
        reason = info.get("overageDisabledReason") or info.get("status", "rate limited")
        pieces = [str(reason).replace("_", " ")]
        if resets_at:
            pieces.append(f"resetsAt={resets_at}")
        self._print_line("rate limit", "  ".join(pieces), "red")

    def _print_tool(self, name: str, tool_input: dict):
        summary = self._summarize_tool_input(tool_input)
        table = Table.grid(expand=False)
        table.add_column(style="bold magenta", no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column(style="white")
        table.add_row(name, "  ", summary)
        self.console.print(Panel(table, title="tool", border_style="magenta"))

    def _print_line(self, label: str, text: str, style: str):
        label_text = Text(f"{label:>10}", style=f"bold {style}")
        body = Text(text or "", style=style)
        self.console.print(Group(label_text, body), soft_wrap=True)

    def _print_json(self, label: str, payload: dict):
        syntax = Syntax(
            json.dumps(payload, indent=2, sort_keys=True),
            "json",
            theme="ansi_dark",
            word_wrap=True,
        )
        self.console.print(Panel(syntax, title=label, border_style="yellow"))

    def _parse_tool_json(self, raw_json: str) -> dict:
        if not raw_json:
            return {}
        try:
            parsed = json.loads(raw_json)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except json.JSONDecodeError:
            return {"input": raw_json}

    def _summarize_tool_input(self, tool_input: dict) -> str:
        for key in ("file_path", "path", "notebook_path"):
            if tool_input.get(key):
                return str(tool_input[key])
        if tool_input.get("command"):
            return str(tool_input["command"]).replace("\n", " && ")
        if tool_input.get("pattern"):
            path = tool_input.get("path") or tool_input.get("glob") or ""
            return f"{tool_input['pattern']} {path}".strip()
        if tool_input:
            return json.dumps(tool_input, sort_keys=True)
        return ""
