# 복잡한 워크플로우
# 장점 : 상태 관리, 분기 처리

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline  # LangChain 0.0.37 이후로 HuggingFacePipeline이 별도 패키지로 분리

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from typing import TypedDict

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

# 3. 파이프라인 생성
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

# 4. LangChain 파이프라인 연결
slm = HuggingFacePipeline(pipeline=pipe)


# 5. 상태 정의
class AgentState(TypedDict):
    input: str
    response: str
    
# 6. 노드 정의
def generate_response(state: AgentState):
    enhanced_prompt = f"[INST] {state['input']} [/INST]"
    return {"response": slm.invoke(enhanced_prompt)}

def validate_response(state: AgentState):
    # 응답 검증 로직 (예: 특정 키워드 확인)
    if "error" in state["response"].lower():
        return {"input": "다시 시도해주세요."}
    return state

# 7. 그래프 구성
workflow = StateGraph(AgentState)
workflow.add_node("generator", generate_response)
workflow.add_node("validator", validate_response)

# END 노드 추가
workflow.add_node("END", lambda state: state)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "validator")
workflow.add_conditional_edges(
    "validator",
    lambda x: "재시도" if "다시" in x["response"] else "종료",
    {"종료": "END", "재시도": "generator"}
)

# 8. 실행
app = workflow.compile()
result = app.invoke({"input": "파리 여행 팁을 알려주세요"})
print(result["response"])