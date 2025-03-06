from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from threading import Thread

# 1. 모델 및 토크나이저 로드
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cache_dir = 'open-framework/'

tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    cache_dir=cache_dir
)

# 2. CoT 프롬프트 (단계별 출력 유도) <- Zero-shot Learning 
prompt = input(r"질문해주세요 : ")
cot_prompt = f"""<|system|>
당신은 논리적 사고가 뛰어난 전문가입니다. 다음 단계로 답변하세요:
1. [분석] 사용자의 요구사항 정리
2. [수집] 관련 정보 종합
3. [추천] 근거를 포함한 최종 답변
</s>
<|user|>
{prompt}
</s>
<|assistant|>
"""

# 3. 실시간 스트리머 설정
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 4. 파이프라인 구성
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.1,
    streamer=streamer  # 스트리머 주입
)

# 5. 별도 스레드에서 생성 실행
generation_thread = Thread(target=pipe, kwargs={"text_inputs": cot_prompt})
generation_thread.start()

# 6. 실시간 출력 확인 (스트리머가 자동 처리)
print("실시간 추론 과정:\n" + "="*40)
generation_thread.join()