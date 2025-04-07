import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)

class SolarChatbot:
    def __init__(self, name: str = "Solar Assistant", memory_limit: int = 10):
        """초기화 함수
        
        Args:
            name (str): 챗봇의 이름
            memory_limit (int): 메모리에 저장할 최대 대화 수
        """
        self.name = name
        self.memory: List[Dict[str, Any]] = []
        self.memory_limit = memory_limit
        
        # 모델 초기화
        self.model_id = "upstage/TinySolar-248m-4k-py-instruct"
        self.cache_dir = './cache'
        
        try:
            logging.info("토크나이저 로딩 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            logging.info("모델 로딩 중...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                cache_dir=self.cache_dir
            )
            
            logging.info(f"{self.name} 초기화 완료")
        except Exception as e:
            logging.error(f"모델 초기화 중 오류 발생: {str(e)}")
            raise
    
    def _manage_memory(self):
        """메모리 관리 함수"""
        if len(self.memory) > self.memory_limit:
            removed = self.memory[:-self.memory_limit]
            self.memory = self.memory[-self.memory_limit:]
            logging.info(f"{len(removed)}개의 오래된 대화가 메모리에서 제거되었습니다.")
    
    def _format_prompt(self, user_input: str) -> str:
        """프롬프트 포맷팅 함수
        
        Args:
            user_input (str): 사용자 입력
            
        Returns:
            str: 포맷팅된 프롬프트
        """
        # 이전 대화 기록을 포함한 프롬프트 생성
        memory_str = "\n".join([
            f"User: {m['input']}\nAssistant: {m['response']}"
            for m in self.memory[-3:]  # 최근 3개의 대화만 포함
        ])
        
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
        )
        
        if memory_str:
            prompt += f"Previous conversation:\n{memory_str}\n\n"
            
        prompt += f"### Instruction:\n{user_input}\n\n### Response:\n"
        
        return prompt
    
    def generate_response(self, user_input: str, max_length: int = 512) -> str:
        """응답 생성 함수
        
        Args:
            user_input (str): 사용자 입력
            max_length (int): 최대 토큰 길이
            
        Returns:
            str: 생성된 응답
        """
        try:
            # 프롬프트 포맷팅
            prompt = self._format_prompt(user_input)
            
            # 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 응답 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 응답 디코딩 및 정제
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # 메모리에 대화 저장
            self.memory.append({
                "input": user_input,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # 메모리 관리
            self._manage_memory()
            
            return response
            
        except Exception as e:
            error_msg = f"응답 생성 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            return error_msg
    
    def chat(self):
        """대화 인터페이스 실행"""
        print(f"{self.name} 초기화 완료! 종료하려면 'quit' 입력")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                response = this.generate_response(user_input)
                print("\nAssistant:", response)
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {str(e)}")
                continue

def main():
    try:
        # 챗봇 인스턴스 생성
        chatbot = SolarChatbot(
            name="Solar Assistant",
            memory_limit=10
        )
        
        # 대화 시작
        chatbot.chat()
        
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 