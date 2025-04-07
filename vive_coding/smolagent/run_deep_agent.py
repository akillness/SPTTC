#!/usr/bin/env python3
"""
DeepAgent 실행 예제 스크립트
이 스크립트는 DeepAgent 클래스를 사용하여 다양한 작업을 수행하는 방법을 보여줍니다.
"""

import logging
from deep_agent import DeepAgent

def main():
    """메인 함수"""
    try:
        # 에이전트 생성
        agent = DeepAgent(
            name="딥도우미",
            description="저는 여러분의 질문에 답하고 작업을 도와주는 AI 에이전트입니다. 한국어로 질문하면 한국어로, 영어로 질문하면 영어로 답변합니다.",
            memory_limit=5  # 메모리 제한을 5개로 설정
        )
        
        # 대화형 모드
        print("=" * 50)
        print("DeepAgent 대화형 모드")
        print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
        print("=" * 50)
        
        while True:
            # 사용자 입력 받기
            user_input = input("\n질문을 입력하세요: ")
            
            # 종료 조건 확인
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("프로그램을 종료합니다.")
                break
            
            # 작업 실행
            result = agent.run_task(user_input)
            print(f"\n결과: {result}")
            print("-" * 50)
            
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 