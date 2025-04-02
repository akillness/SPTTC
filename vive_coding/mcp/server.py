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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = f"https://www.google.com/search?q={city}+weather&hl=ko"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        if response.status_code != 200:
            return f"Error: Unable to fetch weather for {city}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        try:
            # 새로운 클래스명으로 날씨 정보 파싱
            weather_div = soup.select_one('#wob_wc')
            if not weather_div:
                return f"날씨 정보를 찾을 수 없습니다: {city}"

            temp = soup.select_one('#wob_ttm')
            temp = temp.text if temp else "N/A"
            
            condition = soup.select_one('#wob_dcp')
            condition = condition.text if condition else "N/A"
            
            humidity = soup.select_one('#wob_hm')
            humidity = humidity.text if humidity else "N/A"
            
            precipitation = soup.select_one('#wob_pp')
            precipitation = precipitation.text if precipitation else "N/A"

            return (
                f"🌤️ {city} 날씨 안내\n"
                f"▫️ 현재 상태: {condition}\n"
                f"▫️ 기온: {temp}°C\n"
                f"▫️ 습도: {humidity}\n"
                f"▫️ 강수확률: {precipitation}"
            )
        except Exception as e:
            return f"날씨 정보를 파싱하는 중 오류가 발생했습니다: {str(e)}"

@mcp.tool()
def generate_address(count: int = 1, locale: str = None) -> str:
    """Generate address data including street, city, state, country, and coordinates"""
    # Implementation of generate_address function
    pass

