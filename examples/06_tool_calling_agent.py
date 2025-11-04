"""
Example 6: Automatic Tool Calling

Demonstrates:
- Registering tools with the @tool decorator
- Automatic tool detection based on user input
- Tool execution and result integration
- Agentic loop (tool calling until done)

The AI automatically detects when it needs a tool and calls it!

User: "What's 25 * 17?"
 → AI detects it needs calculator tool
 → Calls calculator(25, 17, "multiply")
 → Returns result to user
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.context_engine import Agent, AgentConfig, LLMConfig
from src.context_engine.core.tools import tool

# Load environment variables
load_dotenv()


# ============================================================================
# Define Tools
# ============================================================================

@tool(description="Get the current weather for a city")
def get_weather(city: str, units: str = "celsius") -> dict:
    """
    Get weather information for a city.

    Args:
        city: City name
        units: Temperature units (celsius or fahrenheit)

    Returns:
        Weather data dictionary
    """
    # Simulated weather data
    weather_db = {
        "paris": {"temp": 18, "condition": "Cloudy", "humidity": 65},
        "london": {"temp": 15, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp": 22, "condition": "Sunny", "humidity": 55},
        "new york": {"temp": 20, "condition": "Partly Cloudy", "humidity": 60},
        "sydney": {"temp": 25, "condition": "Sunny", "humidity": 50},
    }

    city_lower = city.lower()
    if city_lower in weather_db:
        data = weather_db[city_lower].copy()
        if units == "fahrenheit":
            data["temp"] = round(data["temp"] * 9/5 + 32)
            data["units"] = "°F"
        else:
            data["units"] = "°C"
        return data
    else:
        return {"error": f"Weather data not available for {city}"}


@tool(description="Perform mathematical calculations")
def calculator(a: float, b: float, operation: str = "add") -> dict:
    """
    Perform basic mathematical operations.

    Args:
        a: First number
        b: Second number
        operation: Operation to perform (add, subtract, multiply, divide)

    Returns:
        Calculation result
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None,
    }

    if operation in operations:
        result = operations[operation](a, b)
        if result is not None:
            return {
                "operation": operation,
                "a": a,
                "b": b,
                "result": result,
                "expression": f"{a} {operation} {b} = {result}"
            }
        else:
            return {"error": "Division by zero"}
    else:
        return {"error": f"Unknown operation: {operation}"}


@tool(description="Search for information (simulated)")
def web_search(query: str, num_results: int = 3) -> dict:
    """
    Search the web for information (simulated).

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Search results
    """
    # Simulated search results
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i+1} for '{query}'",
                "snippet": f"This is a simulated search result snippet about {query}...",
                "url": f"https://example.com/result{i+1}"
            }
            for i in range(num_results)
        ]
    }


@tool(description="Convert currency between different types")
def currency_converter(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert currency from one type to another.

    Args:
        amount: Amount to convert
        from_currency: Source currency code (USD, EUR, GBP, JPY)
        to_currency: Target currency code

    Returns:
        Conversion result
    """
    # Simulated exchange rates (relative to USD)
    rates = {
        "USD": 1.0,
        "EUR": 0.85,
        "GBP": 0.73,
        "JPY": 110.0,
        "AUD": 1.35,
    }

    if from_currency not in rates or to_currency not in rates:
        return {"error": "Unsupported currency"}

    # Convert to USD first, then to target currency
    usd_amount = amount / rates[from_currency]
    converted = usd_amount * rates[to_currency]

    return {
        "from": f"{amount} {from_currency}",
        "to": f"{round(converted, 2)} {to_currency}",
        "rate": round(converted / amount, 4),
        "converted_amount": round(converted, 2)
    }


# ============================================================================
# Main Example
# ============================================================================

def main():
    """Run tool calling agent example."""
    print("=" * 70)
    print("Example 6: Automatic Tool Calling")
    print("=" * 70)
    print()
    print("This demonstrates AI agents that automatically detect when they")
    print("need to use tools and call them without explicit instructions!")
    print()

    # Configure LLM
    llm_config = LLMConfig(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
    )

    # Configure agent WITH TOOLS ENABLED
    agent_config = AgentConfig(
        name="ToolAgent",
        role="helpful assistant with tools",
        system_prompt=(
            "You are a helpful AI assistant with access to tools. "
            "Use tools when appropriate to provide accurate information. "
            "When you need weather info, use get_weather. "
            "For calculations, use calculator. "
            "For web searches, use web_search. "
            "For currency conversion, use currency_converter."
        ),
        llm_config=llm_config,
        enable_tools=True,  # ← KEY: Enable automatic tool calling
        max_tool_iterations=5,
    )

    # Create agent
    print("Creating agent with tools...")
    agent = Agent(agent_config)

    # Register tools
    print("Registering tools:")
    print("  • get_weather - Get weather for cities")
    print("  • calculator - Perform math operations")
    print("  • web_search - Search the web (simulated)")
    print("  • currency_converter - Convert currencies")
    print()

    agent.add_tool_function(get_weather)
    agent.add_tool_function(calculator)
    agent.add_tool_function(web_search)
    agent.add_tool_function(currency_converter)

    print(f"✓ {len(agent.tool_registry)} tools registered")
    print()

    # Test automatic tool calling
    print("=" * 70)
    print("DEMONSTRATION: Automatic Tool Detection and Calling")
    print("=" * 70)
    print()

    # Example 1: Weather query
    print("Example 1: Weather Query")
    print("-" * 70)
    query1 = "What's the weather like in Paris?"
    print(f"User: {query1}")
    print()
    response1 = agent.process_message(query1)
    print(f"\nAssistant: {response1}")
    print()
    print("-" * 70)
    print()

    # Example 2: Math calculation
    print("Example 2: Mathematics")
    print("-" * 70)
    query2 = "Can you calculate 127 * 43 for me?"
    print(f"User: {query2}")
    print()
    response2 = agent.process_message(query2)
    print(f"\nAssistant: {response2}")
    print()
    print("-" * 70)
    print()

    # Example 3: Currency conversion
    print("Example 3: Currency Conversion")
    print("-" * 70)
    query3 = "How much is 100 USD in EUR?"
    print(f"User: {query3}")
    print()
    response3 = agent.process_message(query3)
    print(f"\nAssistant: {response3}")
    print()
    print("-" * 70)
    print()

    # Example 4: Complex query (multiple tools)
    print("Example 4: Complex Query (Multiple Tools)")
    print("-" * 70)
    query4 = "What's the weather in Tokyo and London? Also calculate which city is warmer by how many degrees."
    print(f"User: {query4}")
    print()
    response4 = agent.process_message(query4)
    print(f"\nAssistant: {response4}")
    print()
    print("-" * 70)
    print()

    # Show statistics
    print("=" * 70)
    print("Agent Statistics")
    print("=" * 70)
    stats = agent.get_stats()
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Total tokens used: {stats['total_tokens']}")
    print(f"Tools available: {len(agent.tool_registry)}")
    print()

    # Explain how it works
    print("=" * 70)
    print("How It Works")
    print("=" * 70)
    print()
    print("1. User asks a question")
    print("2. Agent receives question + tool definitions")
    print("3. LLM decides: 'Do I need a tool for this?'")
    print("   → YES: Returns tool call request")
    print("   → NO: Returns text response")
    print("4. Framework executes tool automatically")
    print("5. Tool result is added to conversation")
    print("6. LLM sees result and formulates final answer")
    print("7. User gets response with tool-enhanced information")
    print()
    print("This is TRUE agentic behavior - the AI decides when to use tools!")
    print()

    # Key benefits
    print("=" * 70)
    print("Key Benefits")
    print("=" * 70)
    print()
    print("✓ Automatic tool selection - AI decides when to use tools")
    print("✓ Multi-step reasoning - Can use multiple tools in sequence")
    print("✓ Error handling - Failed tool calls don't crash the agent")
    print("✓ Transparent - See exactly what tools are being called")
    print("✓ Extensible - Easy to add new tools")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()
