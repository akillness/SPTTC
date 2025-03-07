# 문서 기반 QA, RAG
# 장점 : 검색 최적화, 쉬운 문서 통합

import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PyMuPDFReader,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer,AutoModelForCausalLM


'''
from llama_index.core.readers.base import BaseReader

# 1. 커스텀 파서 클래스 정의
class MyCSVReader(BaseReader):
    def load_data(self, file_path):
        import pandas as pd
        df = pd.read_csv(file_path)
        return [Document(text=df.to_string())]

# 2. SimpleDirectoryReader에 파서 등록
reader = SimpleDirectoryReader(
    input_dir="data/",
    file_extractor={".csv": MyCSVReader()}
)
documents = reader.load_data()

'''
# 0. 임베딩 모델 설정
# 0.1 전역 설정에 임베딩 모델 지정
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # model_name="jhgan/ko-sroberta-multitask" 
    # sentence-transformers/all-mpnet-base-v2" 
    # model_kwargs={"device": "cuda"}
)


# 1. 모델을 로컬 캐시 폴더에 다운로드
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # < 영어전용
# model_name = "EleutherAI/polyglot-ko-1.3b" # < 한국어전용;;
cache_dir = "open-framework"  # 사용자 지정 캐시 경로

'''
# 모델 다운로드 (최초 1회 실행)
snapshot_download(model_name, cache_dir=cache_dir)

# 2. 로컬 경로에서 모델 로드
local_model_path = f"{cache_dir}/models--{model_name.replace('/', '--')}/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6" 
# local_model_path = f"{cache_dir}/models--{model_name.replace('/', '--')}/snapshots/557e162cf6e944fdbae05bab2e45d066a125eacb" 
'''
''' EleutherAI/polyglot-ko-1.3b 인 경우 수정사항, '''

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
)

# 3. 모델 초기화 (LLamaIndex 전용 인터페이스)
# 3.1 전역 설정에 LLM 지정
Settings.llm = HuggingFaceLLM( # ✅ 핵심 수정
    # model_name=local_model_path,
    # tokenizer_name=local_model_path,
    model=model,
    tokenizer=tokenizer,
    context_window=2048,
    max_new_tokens=512,
    device_map="auto",
    generate_kwargs= { 
                      "do_sample":True, 
                      "temperature": 0.3, 
                      "repetition_penalty": 1.2,
                      "pad_token_id": tokenizer.eos_token_id,
                    # "pad_token_id": tokenizer.eos_token_id  # 패딩 토큰 설정 추가
                    },
)


# 4. 프롬프트 템플릿 설정
qa_template = PromptTemplate(
    """
    <|user|>
    {query_str}
    </s>
    <|assistant|>
    """
)
# 5. 문서 로드 및 인덱스 생성
data_dir = 'open-framework/data/'

doc_parser = DocxReader()
hwp_parser = HWPReader()
pdf_parser = PyMuPDFReader()
file_extractor = {".docx": doc_parser,
                  ".txt": hwp_parser,
                  ".pdf": pdf_parser,
                  }

documents = SimpleDirectoryReader( #txt 파일만 포함된 폴더 전체를 불러오기 : SimpleDirectoryReader
    data_dir, 
    file_extractor=file_extractor,
    encoding="utf-8",
    # recursive=False,
    # required_exts=[".txt", ".pdf"],
    # exclude=["temp/"],
).load_data()

# for d in documents:
#     print(d)

index = VectorStoreIndex.from_documents(documents,show_progress=True)

# 6. 쿼리 엔진 실행
query_engine = index.as_query_engine(text_qa_template=qa_template,streaming=True) #질의와 관련된 청크 추출

query = input("질문해주세요 : ")
response = query_engine.query(query) # LLM에 전달

# 스트리밍 출력
print("답변 스트리밍 시작:", end="\n\n", flush=True)
response_text = ""
for token in response.response_gen:
    response_text += token
    print(token, end="", flush=True)  # 토큰 단위로 실시간 출력

# 최종 완료된 응답 확인 (선택 사항)
print("\n\n=== 완성된 응답 ===")
print(response_text)

'''
- 의존성 관리: PDF나 DOCX를 읽으려면 pip install pypdf python-docx 필요.
- 대용량 파일: 메모리 부족 방지를 위해 청크 단위 처리 필요 (별도 구성).
- 이미지 처리: OCR 사용 시 pytesseract와 Tesseract 엔진 설치 필수.
'''