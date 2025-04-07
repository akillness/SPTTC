#!/usr/bin/env python3
"""
에이전트 비교 스크립트
이 스크립트는 SmolAgent와 DeepAgent를 비교하여 동일한 작업에 대한 응답을 보여줍니다.
"""

import logging
import time
from simple_agent import SmolAgent
from deep_agent import DeepAgent

def run_comparison(task, smol_agent, deep_agent):
    """두 에이전트를 비교하여 동일한 작업에 대한 응답을 보여줍니다.
    
    Args:
        task (str): 실행할 작업
        smol_agent (SmolAgent): SmolAgent 인스턴스
        deep_agent (DeepAgent): DeepAgent 인스턴스
    """
    print(f"\n작업: {task}")
    print("-" * 50)
    
    # SmolAgent 실행
    print("SmolAgent 응답:")
    start_time = time.time()
    smol_result = smol_agent.run_task(task)
    smol_time = time.time() - start_time
    print(f"결과: {smol_result}")
    print(f"소요 시간: {smol_time:.2f}초")
    print("-" * 50)
    
    # DeepAgent 실행
    print("DeepAgent 응답:")
    start_time = time.time()
    deep_result = deep_agent.run_task(task)
    deep_time = time.time() - start_time
    print(f"결과: {deep_result}")
    print(f"소요 시간: {deep_time:.2f}초")
    print("-" * 50)
    
    # 결과 비교
    print("결과 비교:")
    print(f"SmolAgent 소요 시간: {smol_time:.2f}초")
    print(f"DeepAgent 소요 시간: {deep_time:.2f}초")
    print(f"시간 차이: {abs(smol_time - deep_time):.2f}초")
    print("=" * 50)

def main():
    """메인 함수"""
    try:
        # 에이전트 생성
        smol_agent = SmolAgent(
            name="도우미",
            description="저는 여러분의 질문에 답하고 작업을 도와주는 AI 에이전트입니다.",
            memory_limit=5
        )
        
        deep_agent = DeepAgent(
            name="딥도우미",
            description="저는 여러분의 질문에 답하고 작업을 도와주는 AI 에이전트입니다. 한국어로 질문하면 한국어로, 영어로 질문하면 영어로 답변합니다.",
            memory_limit=5
        )
        
        # 테스트 작업 목록
        tasks = [
            "2023년 노벨 물리학상 수상자는 누구인가요?",  # 검색 기능 테스트
            "123 * 456은 얼마인가요?",  # 계산 기능 테스트
            "이 숫자들의 합을 구해주세요: 1, 2, 3, 4, 5",  # 숫자 합계 테스트
            "인공지능의 발전이 현대 사회에 미치는 영향을 3줄로 요약해주세요.",  # 일반 응답 테스트
            "What is the future of AI technology?",  # 영어 응답 테스트
        ]
        
        # 각 작업에 대해 두 에이전트 비교 실행
        for task in tasks:
            run_comparison(task, smol_agent, deep_agent)
            
        # 대화형 모드
        print("\n대화형 모드로 전환합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
        print("에이전트를 선택하려면 'smol' 또는 'deep'을 입력하세요.")
        
        while True:
            # 사용자 입력 받기
            user_input = input("\n질문을 입력하세요: ")
            
            # 종료 조건 확인
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("프로그램을 종료합니다.")
                break
            
            # 에이전트 선택
            agent_choice = input("어떤 에이전트를 사용하시겠습니까? (smol/deep): ").lower()
            
            if agent_choice == "smol":
                result = smol_agent.run_task(user_input)
                print(f"\nSmolAgent 결과: {result}")
            elif agent_choice == "deep":
                result = deep_agent.run_task(user_input)
                print(f"\nDeepAgent 결과: {result}")
            else:
                print("잘못된 선택입니다. 'smol' 또는 'deep'을 입력하세요.")
            
            print("-" * 50)
            
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 