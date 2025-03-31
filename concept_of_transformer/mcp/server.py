# server.py
import httpx
from mcp.server.fastmcp import FastMCP
from bs4 import BeautifulSoup

# Create an MCP server
mcp = FastMCP("My App")

# Add an addition tool
@mcp.tool()
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI using weight(kg) and height(cm)"""
    height_m = height_cm / 100  # cm → m 변환 추가
    return round(weight_kg / (height_m ** 2), 2)  # 소수점 둘째자리 반올림

@mcp.tool()
async def get_weather(city: str) -> str:
    """Get weather information for a specific city using Google Weather"""
    # 국가명 입력 시 수도로 변환
    if city.lower() in ["korea", "대한민국", "한국"]:
        city = "Seoul"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    url = f"https://www.google.com/search?q=weather+{city}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        if response.status_code != 200:
            return f"Error: Unable to fetch weather for {city}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 추가 정보 추출 (강수 확률, 습도)
        details = soup.find('div', class_='BNeawe s3v9rd AP7Wnd')
        detail_text = details.text if details else ""
        
        # 기존 온도 및 상태 추출
        temp_element = soup.find('div', class_='BNeawe iBp4i AP7Wnd')
        condition_element = soup.find('div', class_='BNeawe tAd8D AP7Wnd')
        
        if not temp_element or not condition_element:
            return f"Error: Could not parse weather data for {city}"
            
        temp = temp_element.text.split('\n')[0]
        condition = condition_element.text.split('\n')[1]
        
        return (
            f"🇰🇷 {city} 날씨 안내\n"
            f"▫️ 현재 상태: {condition}\n"
            f"▫️ 기온: {temp}\n"
            f"▫️ 추가 정보: {detail_text.replace(' · ', ', ')}"
        )

@mcp.tool()
def generate_address(count: int = 1, locale: str = None) -> str:
    """Generate address data including street, city, state, country, and coordinates"""
    # Implementation of generate_address function
    pass

