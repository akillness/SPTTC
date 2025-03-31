# server.py
import httpx
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("My App")

# Add an addition tool
@mcp.tool()
def calculate_bmi(wieght_kg: float, height_cm:float) -> float:      
    """Calculate BMI""" 
    return wieght_kg / (height_cm **2 )

@mcp.tool()
async def get_weather(city: str) -> str:
    """Get weather information for a specific city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text

