"""Auditable tool-routing demo with explicit allow-listed tools."""
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class ToolResult:
    tool: str
    output: str

def utc_time(_: str) -> ToolResult:
    return ToolResult("utc_time", datetime.now(timezone.utc).isoformat())

def echo(value: str) -> ToolResult:
    return ToolResult("echo", value)

TOOLS = {"utc_time": utc_time, "echo": echo}

def run(tool: str, argument: str = "") -> ToolResult:
    if tool not in TOOLS:
        raise ValueError(f"Tool not allowed: {tool}")
    result = TOOLS[tool](argument)
    print({"tool": result.tool, "output": result.output})
    return result

if __name__ == "__main__":
    run("utc_time")
    run("echo", "Agent workflow is explicit and auditable.")
