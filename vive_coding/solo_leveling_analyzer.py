import os
import json
import time
import sys
from typing import Dict, List
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Google AI 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

class SoloLevelingAnalyzer:
    """
    '나 혼자만 레벨업' 웹소설/웹툰의 시나리오를 분석하는 에이전트
    """
    
    def __init__(self):
        # 모델 설정
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1500,
        }
        
        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-8b-exp-0924",
            generation_config=self.generation_config
        )
    
    def analyze_scenario(self, scenario_name: str) -> Dict:
        """
        시나리오의 게임 요소를 분석합니다.
        """
        prompt = f"""
당신은 '나 혼자만 레벨업' 웹소설/웹툰의 게임 요소를 전문적으로 분석하는 전문가입니다.
다음 시나리오에 대해 게임 디자인 관점에서 상세히 분석해주세요: {scenario_name}

다음 형식으로 JSON 응답을 제공해주세요:

===분석 데이터===
{{
    "scenario_name": "시나리오 이름",
    "game_elements": {{
        "character_system": {{
            "level_system": "레벨링 시스템 설명",
            "stats": ["주요 스탯 목록"],
            "skills": ["주요 스킬 목록"]
        }},
        "dungeon_system": {{
            "types": ["던전 종류"],
            "mechanics": ["주요 던전 메카닉"]
        }},
        "quest_system": {{
            "types": ["퀘스트 종류"],
            "rewards": ["보상 종류"]
        }},
        "item_system": {{
            "equipment": ["장비 종류"],
            "consumables": ["소모품 종류"]
        }}
    }},
    "unique_features": ["특별한 게임적 요소들"]
}}

===상세 설명===
1. 게임 시스템으로서의 특징
2. 밸런싱 요소 분석
3. 흥미로운 메카닉 설명

위 형식을 정확히 지켜서 분석해주세요. 특히 JSON 형식을 정확하게 지켜주세요.
"""
        response = self.model.generate_content(prompt)
        
        # 응답 파싱
        sections = response.text.split("===")
        analysis_data = {}
        detailed_explanation = ""
        
        found_data_section = False
        found_explanation_section = False
        
        for i, section in enumerate(sections):
            section_content = section.strip()
            
            # 분석 데이터 섹션 찾기
            if "분석 데이터" in section_content:
                found_data_section = True
                continue
                
            # 상세 설명 섹션 찾기
            if "상세 설명" in section_content:
                found_explanation_section = True
                continue
                
            # 분석 데이터 섹션 다음에 오는 섹션이 JSON 데이터
            if found_data_section and not found_explanation_section:
                # JSON 데이터 추출 및 정제
                json_text = section_content
                
                # 마크다운 코드 블록 제거
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1]
                if "```" in json_text:
                    json_text = json_text.split("```")[0]
                
                try:
                    analysis_data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {str(e)}")
                    print("원본 JSON 텍스트:")
                    print(json_text)
                    # 기본 분석 데이터 생성
                    analysis_data = {
                        "scenario_name": scenario_name,
                        "parse_error": "JSON 파싱 오류가 발생했습니다."
                    }
                
                found_data_section = False  # 처리 완료
                    
            # 상세 설명 섹션 다음에 오는 섹션이 설명 텍스트
            if found_explanation_section:
                detailed_explanation = section_content
                found_explanation_section = False  # 처리 완료
        
        return {
            "analysis": analysis_data,
            "detailed_explanation": detailed_explanation
        }
    
    def print_analysis_report(self, result: Dict):
        """
        분석 결과를 보기 좋게 출력합니다. 스트리머 효과를 적용하여 점진적으로 출력합니다.
        """
        analysis = result.get("analysis", {})
        explanation = result.get("detailed_explanation", "")
        
        self.stream_print("\n===== '나 혼자만 레벨업' 시나리오 분석 =====\n", 0.05)
        
        if "parse_error" in analysis:
            self.stream_print(f"분석 오류: {analysis['parse_error']}", 0.05)
            return
            
        self.stream_print(f"📌 시나리오: {analysis.get('scenario_name', '정보 없음')}\n", 0.07)
        
        # 게임 요소 출력
        game_elements = analysis.get("game_elements", {})
        
        self.stream_print("🎮 캐릭터 시스템", 0.1)
        character_system = game_elements.get("character_system", {})
        self.stream_print(f"  • 레벨 시스템: {character_system.get('level_system', '정보 없음')}", 0.03)
        self.stream_print("  • 주요 스탯:", 0.1)
        for stat in character_system.get("stats", ["정보 없음"]):
            self.stream_print(f"    - {stat}", 0.08)
        self.stream_print("  • 주요 스킬:", 0.1)
        for skill in character_system.get("skills", ["정보 없음"]):
            self.stream_print(f"    - {skill}", 0.08)
        self.stream_print("", 0.5)
        
        self.stream_print("🏰 던전 시스템", 0.1)
        dungeon_system = game_elements.get("dungeon_system", {})
        self.stream_print("  • 던전 종류:", 0.1)
        for dungeon_type in dungeon_system.get("types", ["정보 없음"]):
            self.stream_print(f"    - {dungeon_type}", 0.08)
        self.stream_print("  • 주요 메카닉:", 0.1)
        for mechanic in dungeon_system.get("mechanics", ["정보 없음"]):
            self.stream_print(f"    - {mechanic}", 0.08)
        self.stream_print("", 0.5)
        
        self.stream_print("📜 퀘스트 시스템", 0.1)
        quest_system = game_elements.get("quest_system", {})
        self.stream_print("  • 퀘스트 종류:", 0.1)
        for quest_type in quest_system.get("types", ["정보 없음"]):
            self.stream_print(f"    - {quest_type}", 0.08)
        self.stream_print("  • 보상 종류:", 0.1)
        for reward in quest_system.get("rewards", ["정보 없음"]):
            self.stream_print(f"    - {reward}", 0.08)
        self.stream_print("", 0.5)
        
        self.stream_print("🎒 아이템 시스템", 0.1)
        item_system = game_elements.get("item_system", {})
        self.stream_print("  • 장비 종류:", 0.1)
        for equipment in item_system.get("equipment", ["정보 없음"]):
            self.stream_print(f"    - {equipment}", 0.08)
        self.stream_print("  • 소모품 종류:", 0.1)
        for consumable in item_system.get("consumables", ["정보 없음"]):
            self.stream_print(f"    - {consumable}", 0.08)
        self.stream_print("", 0.5)
        
        self.stream_print("🌟 특별한 게임 요소", 0.1)
        for feature in analysis.get("unique_features", ["정보 없음"]):
            self.stream_print(f"  • {feature}", 0.08)
        self.stream_print("", 0.5)
        
        # 상세 설명 출력
        if explanation:
            self.stream_print("\n📋 상세 분석 및 설명", 0.1)
            # 문장 단위로 스트리밍
            for sentence in explanation.split('. '):
                if sentence:
                    self.stream_print(f"{sentence}.", 0.02)
        
        self.stream_print("\n=======================================\n", 0.05)
    
    def stream_print(self, text: str, speed: float = 0.03):
        """
        텍스트를 스트리밍 효과로 출력합니다.
        
        Args:
            text: 출력할 텍스트
            speed: 각 글자가 출력되는 간격 (초)
        """
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(speed)
        print("")

def main():
    analyzer = SoloLevelingAnalyzer()
    
    # 예시 시나리오
    scenarios = [
        "붉은 문 던전 사건",
        "더블 던전 공략",
        "정지훈의 각성",
        "제주도 개미 던전",
        "일본 S급 던전 공략"
    ]
    
    # 사용자 입력 받기
    print("'나 혼자만 레벨업' 시나리오 분석 에이전트입니다.")
    print("분석할 시나리오를 선택하거나 직접 입력하세요:")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    
    print("0. 직접 입력")
    
    choice = input("\n선택 (0-5): ")
    
    if choice.isdigit():
        choice = int(choice)
        if 1 <= choice <= len(scenarios):
            scenario_name = scenarios[choice-1]
        elif choice == 0:
            scenario_name = input("분석할 시나리오 이름을 입력하세요: ")
        else:
            print("잘못된 선택입니다. 기본 시나리오로 분석합니다.")
            scenario_name = "붉은 문 던전 사건"
    else:
        print("잘못된 입력입니다. 기본 시나리오로 분석합니다.")
        scenario_name = "붉은 문 던전 사건"
    
    print(f"\n'{scenario_name}' 시나리오를 분석합니다... 잠시만 기다려주세요.")
    
    # 애니메이션 효과로 로딩 표시
    for _ in range(5):
        for c in ['|', '/', '-', '\\']:
            sys.stdout.write(f'\r분석 중... {c}')
            sys.stdout.flush()
            time.sleep(0.1)
    
    # 시나리오 분석
    result = analyzer.analyze_scenario(scenario_name)
    
    # 로딩 완료 표시
    sys.stdout.write('\r분석 완료!     \n')
    
    # 분석 결과 출력
    analyzer.print_analysis_report(result)

if __name__ == "__main__":
    main() 