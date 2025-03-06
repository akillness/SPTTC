# 문서 기반 QA, RAG
# 장점 : 검색 최적화, 쉬운 문서 통합

import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer,AutoModelForCausalLM

# 0. 임베딩 모델 설정
# 0.1 전역 설정에 임베딩 모델 지정
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # model_name="jhgan/ko-sroberta-multitask"
)

'''
# 한국어 최적화 모델
HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# GPU 가속화
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cuda"}
)

# Local 에 저장된 임베딩 사용
HuggingFaceEmbeddings(model_name="./local_models/all-MiniLM-L6-v2")
'''


# 1. 모델을 로컬 캐시 폴더에 다운로드
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # < 영어전용
# model_name = "EleutherAI/polyglot-ko-1.3b" # < 한국어전용;;
cache_dir = "open-framework"  # 사용자 지정 캐시 경로

# 모델 다운로드 (최초 1회 실행)
snapshot_download(model_name, cache_dir=cache_dir)

# 2. 로컬 경로에서 모델 로드
local_model_path = f"{cache_dir}/models--{model_name.replace('/', '--')}/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6" 
# local_model_path = f"{cache_dir}/models--{model_name.replace('/', '--')}/snapshots/557e162cf6e944fdbae05bab2e45d066a125eacb" 

''' EleutherAI/polyglot-ko-1.3b 인 경우 수정사항, '''
'''
### 1. 커스텀 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,  # 반드시 False로 설정
    return_token_type_ids=False,  # ✅ 핵심 수정
    cache_dir=cache_dir,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # GPU 자동 할당
    cache_dir=cache_dir,
    trust_remote_code=True,
)'''

# 3. 모델 초기화 (LLamaIndex 전용 인터페이스)
# 3.1 전역 설정에 LLM 지정
Settings.llm = HuggingFaceLLM( # ✅ 핵심 수정
    model_name=local_model_path,
    tokenizer_name=local_model_path,
    # model=model,
    # tokenizer=tokenizer,
    context_window=2048,
    max_new_tokens=256,
    device_map="auto",
    generate_kwargs= { 
                      "do_sample":True, 
                      "temperature": 0.3, 
                      "repetition_penalty": 1.2,
                    #   "pad_token_id": tokenizer.eos_token_id,
                    },
)


# 4. 프롬프트 템플릿 설정
qa_template = PromptTemplate(
    "[INST] 다음 질문에 대해 간결하게 답변하세요: {query_str} [/INST]"
)

# 5. 문서 로드 및 인덱스 생성
data_dir = 'open-framework/data/'
documents = SimpleDirectoryReader(data_dir).load_data()  # 사용자 문서 폴더
index = VectorStoreIndex.from_documents(documents)

# 6. 쿼리 엔진 실행
query_engine = index.as_query_engine(text_qa_template=qa_template)
response = query_engine.query("프로젝트 한가지 알려줘")
print(response.response)