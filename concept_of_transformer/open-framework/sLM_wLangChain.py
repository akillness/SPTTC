
# 기본 LLM 통합
# 장점 : 빠른

from langchain_huggingface import HuggingFacePipeline  # LangChain 0.0.37 이후로 HuggingFacePipeline이 별도 패키지로 분리
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. 로컬 모델 경로 지정 (예: Windows 기준)
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 주로 영어로 학습된 모델 <-> EleutherAI/polyglot-ko-1.3b  
# model_path = "EleutherAI/polyglot-ko-1.3b" # <- 잘안됨;;
cache_dir = 'open-framework/'

# 2. 토크나이저 & 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # GPU 자동 할당
    cache_dir=cache_dir
)

# 3. 채팅 템플릿 적용
prompt = """<|user|>
파리 추천 여행지 2곳 해줘
<|assistant|>
"""
# prompt = "파리 여행지 추천 2 곳 해줘"


# 4. 파이프라인 생성
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.5
)

# 5. LangChain 파이프라인 연결
slm = HuggingFacePipeline(pipeline=pipe)

# 간단한 질의응답
response = slm.invoke(prompt)

print(response)  # "파리 여행 시 지하철을 활용하시면 편리합니다..."


'''
# 1. 기존 패키지 제거 (충돌 방지)
pip uninstall langchain

# 2. 새로운 패키지 설치
pip install langchain-huggingface>=0.0.1

# 3. model 의존성 확인, langchain-huggingface >= 0.0.1
pip list | grep -E "langchain|huggingface|transformers"
'''