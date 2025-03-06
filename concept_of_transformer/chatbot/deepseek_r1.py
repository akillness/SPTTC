
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, EncoderDecoderModel, DynamicCache

import timeit
'''
psutil: 시스템 메모리 정보 조회
pympler: 객체 메모리 사용량 정확 측정
'''
import os, psutil

# from memory_profiler import profile
# import cProfile
#import line_profiler
# ㄴ 사용 방법 : kernprof -l -v test.py 
import platform

os_name = platform.system()
if os_name == "Darwin":
    print("This is macOS.")
elif os_name == "Windows":
    print("This is Windows.")
else:
    print("This is another OS:", os_name)
    

# 디바이스 설정 (GPU가 사용 가능하면 GPU, 아니면 CPU)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def timeit_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        # with cProfile.Profile() as pr:
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        print(f"실행 시간 : {execution_time:.4}초, func : {func.__name__} ")
        # pr.print_stats()
        return result
    return wrapper

class deepseek_r1():
    # @profile
    @timeit_decorator
    def __init__(self, model_name, cached_dir, device):
        # 설정 로드
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.sliding_window = None  # config 수정 추가 - sdpa 실행안된다는 오류
        
        '''
        if device == "cpu":
            # 양자화 설정 제거
            if hasattr(config, "quantization_config"):
                delattr(config, "quantization_config")
        '''
        if os_name == "Darwin":
             # 모델 및 토크나이저 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cached_dir,
                config=config, 
                trust_remote_code=True,
                # load_in_4bit=True,            
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                # attn_implementation="flash_attention_2",  # 또는 "eager", use_sdpa=False
                # max_position_embeddings=3000,  # 모델이 지원하는 경우에만 추가
                # device_map="auto",              # 메모리 자동 할당
                torch_dtype=torch.bfloat16  # 메모리 절약을 위해 추가
                )
        elif os_name == "Windows":            
             # 모델 및 토크나이저 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cached_dir,
                config=config, 
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                #attn_implementation="flash_attention_2",  # 또는 "eager", use_sdpa=False
                # max_position_embeddings=3000,  # 모델이 지원하는 경우에만 추가
                torch_dtype=torch.bfloat16  # 메모리 절약을 위해 추가
                )
        elif os_name == "Linux":
            from transformers import BitsAndBytesConfig
            
            # 1. 양자화 설정 객체 생성
            '''
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 8비트는 load_in_8bit=True
                bnb_4bit_quant_type="nf4",       # 양자화 알고리즘 타입 (nf4/fp4)
                bnb_4bit_compute_dtype=torch.float16, # torch.bfloat16 (A100, H100 등)  # 계산 dtype
                bnb_4bit_use_double_quant=True  # 이중 양자화 여부
            )
            '''
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,  # 8비트 변환 임계값
                llm_int8_enable_fp32_cpu_offload=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cached_dir,
                config=config, 
                quantization_config=quantization_config,
                attn_implementation="eager",
                # attn_implementation="flash_attention_2" 해당 방식은 Linux 적합,  # 또는 "eager", cuda 11.7 이상버전에서만 가능 
                # max_position_embeddings=3000,  # 모델이 지원하는 경우에만 추가                
                device_map="auto",              # 메모리 자동 할당                
                )
        
        # RoPE 확장 적용 (Rotary Position Embedding)
        self.model.config.rope_scaling = {
            "type": "dynamic", 
            "factor": 1.5
        }
        
        # decoder 모델 로드시 사용
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cached_dir, trust_remote_code=True)
        # 패딩 토큰 명시적 설정 추가
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 1번 방법 적용
            '''
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 새 패딩 토큰 추가
            self.model.resize_token_embeddings(len(self.tokenizer))  # 모델 임베딩 레이어 조정
            '''
            
        self.device = torch.device(device)
    
    def cleanup(self):
        """모델 리소스 명시적 해제"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("모델 리소스가 성공적으로 해제되었습니다")
        
    @timeit_decorator
    def generate(self,input_text):
        # 입력을 토큰화하고 텐서로 변환
        # device = torch.device(device)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # 모델을 사용하여 출력 생성
        with torch.no_grad():
            # with cProfile.Profile() as pr:
            outputs = self.model.generate(
                **inputs,
                # do_sample=True,  # 샘플링 활성화
                # top_k=50,        # 메모리 사용량 감소
                # top_p=0.95,      # 효율적인 탐색
                # temperature=0.6,  # 출력 다양성 조절
                # max_length=3000,  # 최대 길이 변경
                max_new_tokens=2800,  # 새로운 토큰 생성 제한 (선택사항)
                use_cache=True,          # 캐시 사용으로 성능 향상
                repetition_penalty=1.1,  # 반복 생성 방지
                num_beams=1,              # 빔 서치 비활성화 (메모리 절약)
                pad_token_id=self.tokenizer.eos_token_id  # 패딩 토큰 명시적 지정
            )            
        # pr.print_stats()
        # 출력된 토큰을 문자열로 변환하여 결과 출력
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def get_usable_process_num(self):
        process_num = -1
        self.print_memory_usage()
        if torch.cuda.is_available():
            allocated, max_allocated, reserve = self.get_gpu_memory()
            process_num = int(max_allocated / allocated)
        else:  
            process_num = int(self.get_available_memory() / self.get_cpu_memory())                
        return process_num
    
    def get_available_memory(self):
        # 시스템의 가용 메모리(GB)를 반환합니다.
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 3)  # GB 단위로 반환
    
    def get_cpu_memory(self):
        """현재 프로세스의 CPU 메모리 사용량(GB 단위) 반환"""
        mem_info = self.model.get_memory_footprint()
        return mem_info / (1024 ** 3)  # Bytes -> GB 변환

    def get_gpu_memory(self,device="cuda"):
        """GPU 메모리 사용량(GB 단위) 반환"""        
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # 할당된 메모리
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # 최대 할당된 메모리
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # 예약된 메모리
        return allocated, max_allocated, reserved
    
    def print_memory_usage(self):
        """현재 CPU 및 GPU 메모리 사용량 출력"""
        cpu_mem = self.get_cpu_memory()
        print(f"CPU Memory Usage: {cpu_mem:.2f} GB")

        if torch.cuda.is_available():
            allocated, max_allocated, reserved = self.get_gpu_memory()
            print(f"GPU Memory Usage: {allocated:.2f} GB (Allocated), {max_allocated:.2f} GB (Max), {reserved:.2f} GB (Reserved)")
        else:
            print("No GPU available.")