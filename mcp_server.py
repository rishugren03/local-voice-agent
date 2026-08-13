from mcp.server.mcpserver import MCPServer

mcp = MCPServer("voice-agent-tools")

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '12 * 4 + 7'."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def check_calendar(date: str) -> str:
    """Look up mock calendar availability for a given date, e.g. '2026-08-15'."""
    fake_schedule = {
        "2026-08-15": "Busy: Team standup at 10am, Dentist at 3pm",
        "2026-08-16": "Free all day",
    }
    return fake_schedule.get(date, "No events found for that date.")

if __name__ == "__main__":
    mcp.run()