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

@mcp.tool()
def get_weather(location: str) -> str:
    """Look up mock current weather for a city or place, e.g. 'Seattle, WA'."""
    fake_weather = {
        "seattle": "48F, light rain, wind 8 mph",
        "san francisco": "63F, foggy, wind 10 mph",
        "new york": "71F, sunny, wind 5 mph",
        "london": "55F, overcast, wind 12 mph",
    }
    # Callers pass whatever the schema's example teaches them, so "Seattle, WA" and
    # "seattle" have to land on the same entry. Real weather APIs geocode and
    # tolerate the state/country suffix; the mock matches the city part.
    city = location.strip().lower().split(",")[0].strip()
    return fake_weather.get(city, f"No mock weather data for '{location}'.")

if __name__ == "__main__":
    mcp.run()