
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer

# 1. 모델 초기화
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cache_dir = 'open-framework/'

tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",cache_dir=cache_dir)

# 2. 스트리밍 설정
class CotStreamer(TextStreamer):
    def __init__(self, tokenizer, stage_name):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.stage_name = stage_name
        self.first_token = True
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.first_token:
            print(f"\n\n=== {self.stage_name} 생성 ===")
            self.first_token = False
        print(text, end="", flush=True)


# 3. 단계별 파이프라인 빌더
def build_stage_pipeline(stage_name):
    streamer = CotStreamer(tokenizer, stage_name)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        max_new_tokens=2048,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1,
        streamer=streamer,
        device_map="auto"
    )

# 4. LangChain 러너블 구성
analysis_llm = HuggingFacePipeline(pipeline=build_stage_pipeline("분석"))
research_llm = HuggingFacePipeline(pipeline=build_stage_pipeline("정보 수집"))
answer_llm = HuggingFacePipeline(pipeline=build_stage_pipeline("답변"))

# 5. 프롬프트 템플릿
prompt_templates = {
    "analysis": PromptTemplate.from_template("""
    <|system|>문제 분석가</s>
    <|user|>{question} 이 문제를 2단계로 분해하세요.</s>
    <|assistant|>분석:"""),
    
    "research": PromptTemplate.from_template("""
    <|system|>정보 수집가</s>
    <|user|>문제: {question}
    분석: {analysis}
    관련 정보를 수집하세요.</s>
    <|assistant|>수집:"""),
    
    "answer": PromptTemplate.from_template("""
    <|system|>문제 전문가</s>
    <|user|>문제: {question}
    분석: {analysis}
    수집: {research}
    최종 답변을 생성하세요.</s>
    <|assistant|>답변:""")
}

# 6. CoT 체인 구성
cot_chain = (
    RunnablePassthrough.assign(
        analysis=prompt_templates["analysis"] | analysis_llm
    )
    .assign(
        research=prompt_templates["research"] | research_llm
    )
    .assign(
        answer=prompt_templates["answer"] | answer_llm
    )
)

# 7. 실행
question = input("질문을 입력하세요: ")
print("\n[실시간 추론 시작]")
result = cot_chain.invoke({"question": question})

print("\n\n[최종 정리]")
print(f"# 분석 결과\n{result['analysis']}")
print(f"\n# 수집 정보\n{result['research']}")
print(f"\n# 최종 답변\n{result['answer']}")