from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import torch
from threading import Thread

# 1. 모델 & 토크나이저 로드
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cache_dir = 'open-framework/'

tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir
)

# 2. CoT 프롬프트 템플릿
def build_cot_prompt(question):
    return f"""<|system|>
당신은 게임 시나리오 전문가입니다. 반드시 다음 단계를 따라 답변하세요:
1. 문제를 3개 이하의 키워드로 요약
2. 각 키워드별 핵심 정보 추출
3. 정보를 종합한 최종 결론
</s>
<|user|>
{question}
</s>
<|assistant|>
"""

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 3. 추론 파이프라인 설정
cot_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2,
    num_return_sequences=1,
    streamer = streamer
)

# 4. 질문 생성 및 실행 예제
questions = [
    "MMORPG 마법사와 기사의 stat에 영향을 주는 Balance 상황을 1개 만들어줘",
    # "빗물이 고인 도로에서 자동차가 미끄러지는 이유를 과학적으로 설명하세요.",
    # "삼각형의 두 각이 각각 45도와 90도일 때, 남은 각의 크기와 삼각형 종류는?"
]


for idx, question in enumerate(questions):
    prompt = build_cot_prompt(question)
    # response = cot_pipeline(prompt, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    # 5. 별도 스레드에서 생성 실행
    generation_thread = Thread(target=cot_pipeline, kwargs={"text_inputs": prompt})
    generation_thread.start()
    
    print(f"\n=== 질문 {idx+1} ===")
    print(f"입력: {question}")
    print("\nCoT 추론 과정:")
    # print(response.split("<|assistant|>")[1].strip())  # 생성된 응답만 추출
    print("="*50)
    # 6. 실시간 출력 확인 (스트리머가 자동 처리)
    generation_thread.join()


'''
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 4비트 양자화
    device_map="auto"
)
'''