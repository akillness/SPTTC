from typing import List, Dict, Any
import os
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
import json
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*resume_download.*')
warnings.filterwarnings('ignore', message='Special tokens have been added.*')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent.log')
    ]
)

class DeepAgent:
    def __init__(self, name: str, description: str, memory_limit: int = 10):
        """초기화 함수
        
        Args:
            name (str): 에이전트의 이름
            description (str): 에이전트의 설명
            memory_limit (int, optional): 메모리에 저장할 최대 대화 수. Defaults to 10.
        """
        load_dotenv()  # .env 파일에서 환경 변수 로드
        
        self.name = name
        self.description = description
        self.memory: List[Dict[str, Any]] = []
        self.memory_limit = memory_limit
        
        # 모델 경로 지정
        self.model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.cache_dir = './model_cache'
        
        try:
            # 토크나이저 & 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # 스트리밍 설정
            self.streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 파이프라인 설정
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                streamer=self.streamer,
                return_full_text=False,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )
            
            logging.info(f"{self.name} 에이전트가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logging.error(f"모델 초기화 중 오류 발생: {str(e)}")
            raise
        
        # 시스템 메시지 설정
        self.system_message = f"""당신은 {name}이라는 이름의 이중 언어(한국어/영어) AI 에이전트입니다.
{description}

당신은 다음과 같은 도구들을 사용할 수 있습니다:
1. Calculator - 수학 계산을 수행합니다.
2. Search - 인터넷에서 정보를 검색합니다.
3. Sum - 숫자들의 합을 계산합니다.

각 작업을 수행할 때는 다음과 같은 형식으로 응답해주세요:
1. 먼저 어떤 도구를 사용할지 결정합니다.
2. 도구를 사용하여 정보를 수집합니다.
3. 수집된 정보를 바탕으로 답변을 작성합니다.
"""
                    
    def _manage_memory(self):
        """메모리 관리 함수
        
        오래된 대화를 제거하여 메모리 크기를 제한합니다.
        """
        if len(self.memory) > self.memory_limit:
            removed = self.memory[:-self.memory_limit]
            self.memory = self.memory[-self.memory_limit:]
            logging.info(f"{len(removed)}개의 오래된 대화가 메모리에서 제거되었습니다.")
    
    def _calculate(self, expression: str) -> str:
        """ 안전한 수학 계산을 수행하는 도구
        
        Args:
            expression (str): 계산할 수학 표현식
            
        Returns:
            str: 계산 결과
        """
        try:
            # eval은 위험할 수 있으므로 제한된 환경에서만 실행
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                error_msg = f"유효하지 않은 수학 표현식: {expression}"
                logging.warning(error_msg)
                return error_msg
            
            result = eval(expression)
            logging.info(f"계산 성공: {expression} = {result}")
            return str(result)
        except Exception as e:
            error_msg = f"계산 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            return "계산할 수 없는 표현식입니다."
    
    def _search(self, query: str) -> str:
        """인터넷 검색을 수행하는 도구 (모의 구현)
        
        Args:
            query (str): 검색 쿼리
            
        Returns:
            str: 검색 결과
        """
        try:
            logging.info(f"검색 시도: {query}")
            # 실제 검색 대신 모의 응답
            if "2023년 노벨 물리학상" in query:
                result = "2023년 노벨 물리학상은 Pierre Agostini, Ferenc Krausz, Anne L'Huillier가 공동 수상했습니다. 이들은 아토초 펄스 생성 방법 개발을 통해 전자 동역학 연구에 기여한 공로를 인정받았습니다."
                logging.info("검색 성공")
                return result
            
            logging.warning(f"검색 결과 없음: {query}")
            return "검색 결과를 찾을 수 없습니다."
        except Exception as e:
            error_msg = f"검색 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            return error_msg
    
    def _extract_numbers(self, text: str) -> List[int]:
        """텍스트에서 숫자를 추출하는 도구
        
        Args:
            text (str): 숫자를 추출할 텍스트
            
        Returns:
            List[int]: 추출된 숫자 리스트
        """
        try:
            numbers = re.findall(r'\d+', text)
            result = [int(n) for n in numbers]
            logging.info(f"숫자 추출 성공: {result}")
            return result
        except Exception as e:
            logging.error(f"숫자 추출 중 오류 발생: {str(e)}")
            return []
    
    def _sum_numbers(self, numbers: List[int]) -> str:
        """숫자들의 합을 계산하는 도구
        
        Args:
            numbers (List[int]): 합을 계산할 숫자 리스트
            
        Returns:
            str: 계산 결과
        """
        try:
            if not numbers:
                return "합을 계산할 숫자가 없습니다."
            
            total = sum(numbers)
            logging.info(f"합계 계산 성공: {numbers} = {total}")
            return str(total)
        except Exception as e:
            error_msg = f"합계 계산 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            return error_msg
    
    def _generate_response(self, prompt: str) -> str:
        """DeepSeek 모델을 사용하여 응답 생성
        
        Args:
            prompt (str): 입력 프롬프트
            
        Returns:
            str: 생성된 응답
        """
        try:
            logging.info("응답 생성 중...")
            
            # 프롬프트 포맷팅
            formatted_prompt = f"""<|im_start|>system
{self.system_message}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
            
            # 입력 토큰화
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            # 생성 파라미터 설정
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "max_new_tokens": 512,  # 더 긴 응답을 위해 토큰 수 증가
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": self.streamer
            }
            
            # 스트리밍 처리
            print(f"\n[응답 생성 중] ", end="", flush=True)
            response = ""
            
            # 별도의 스레드에서 생성 실행
            generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=gen_kwargs
            )
            generation_thread.start()
            
            # 스트리밍된 토큰 처리
            for token in self.streamer:
                response += token
                print(token, end="", flush=True)
                
                # <|im_end|> 토큰이 나타나면 생성 중단
                if "<|im_end|>" in response:
                    break
            
            # 응답 정제
            response = response.strip()
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
            
            # 특수 토큰과 불필요한 텍스트 제거
            response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
            response = response.replace("<|start|>", "").replace("<|end|>", "")
            response = response.replace("system", "").replace("user", "").replace("assistant", "")
            response = response.replace("responses:", "").replace("</think>", "")
            
            # 연속된 공백 제거
            response = " ".join(response.split())
            
            print(f"\n응답: {response}\n")
            logging.info("응답 생성 완료")
            
            return response
            
        except Exception as e:
            error_msg = f"응답 생성 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            return error_msg
    
    def run_task(self, task: str) -> str:
        """주어진 작업을 실행
        
        Args:
            task (str): 실행할 작업 설명
            
        Returns:
            str: 작업 실행 결과
        """
        try:
            logging.info(f"작업 시작: {task}")
            
            # 현재 메모리 상태를 문자열로 변환
            memory_str = "\n".join([
                f"- {m['task']}: {m['result']}"
                for m in self.memory
            ])
            
            # 사용자 메시지 생성
            prompt = f"""이전 대화 기록:
{memory_str}

현재 작업: {task}

이 작업을 해결하기 위해 어떻게 하시겠습니까?"""
            
            # DeepSeek 모델로 응답 생성
            gpt_response = self._generate_response(prompt)
            
            # 도구 사용이 필요한 경우 처리
            final_response = None
            
            # 1. 검색 기능 확인
            if "Search" in gpt_response and ("노벨" in task or "Nobel" in task or "2023년 노벨 물리학상" in task):
                logging.info("검색 도구 사용")
                search_result = self._search(task)
                if search_result != "검색 결과를 찾을 수 없습니다.":
                    final_response = search_result
            
            # 2. 계산 기능 확인
            elif "Calculator" in gpt_response and "*" in task:
                logging.info("계산 도구 사용")
                numbers = self._extract_numbers(task)
                if len(numbers) >= 2:
                    result = self._calculate(f"{numbers[0]} * {numbers[1]}")
                    final_response = f"계산 결과: {result}"
            
            # 3. 숫자 합계 계산 확인
            elif "Sum" in gpt_response and ("합" in task or "sum" in task.lower()):
                logging.info("숫자 합계 계산 도구 사용")
                numbers = self._extract_numbers(task)
                if numbers:
                    total = self._sum_numbers(numbers)
                    final_response = f"숫자들의 합: {total}"
            
            # 도구로 해결되지 않은 경우 DeepSeek 모델 사용
            if final_response is None:
                # GPT에게 직접 답변 요청
                follow_up_prompt = f"""이전 대화 기록:
{memory_str}

현재 작업: {task}

이전 응답: {gpt_response}

위 작업에 대해 직접 답변해주세요."""
                
                final_response = self._generate_response(follow_up_prompt)
            
            # 결과를 메모리에 저장
            self.memory.append({
                "task": task,
                "result": final_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # 메모리 관리
            self._manage_memory()
            
            logging.info("작업 완료")
            return final_response
            
        except Exception as e:
            error_msg = f"작업 실행 중 예상치 못한 오류 발생: {str(e)}"
            logging.error(error_msg)
            return error_msg

if __name__ == "__main__":
    try:
        # 에이전트 생성
        agent = DeepAgent(
            name="딥도우미",
            description="저는 여러분의 질문에 답하고 작업을 도와주는 AI 에이전트입니다. 한국어로 질문하면 한국어로, 영어로 질문하면 영어로 답변합니다.",
            memory_limit=5  # 메모리 제한을 5개로 설정
        )
        
        # 테스트 작업 실행
        tasks = [
            "2023년 노벨 물리학상 수상자는 누구인가요?",  # 검색 기능 테스트
            "123 * 456은 얼마인가요?",  # 계산 기능 테스트
            "이 숫자들의 합을 구해주세요: 1, 2, 3, 4, 5",  # 숫자 합계 테스트
            "인공지능의 발전이 현대 사회에 미치는 영향을 3줄로 요약해주세요.",  # 일반 응답 테스트
            "What is the future of AI technology?",  # 영어 응답 테스트
        ]
        
        for task in tasks:
            print(f"\n작업: {task}")
            result = agent.run_task(task)
            print(f"결과: {result}")
            print("-" * 50)
            
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}") 