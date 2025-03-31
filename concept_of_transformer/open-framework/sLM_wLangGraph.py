# 복잡한 워크플로우
# 장점 : 상태 관리, 분기 처리

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline  # LangChain 0.0.37 이후로 HuggingFacePipeline이 별도 패키지로 분리

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from typing import TypedDict, Any
import threading
from transformers import TextIteratorStreamer  # 변경된 스트리머



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

# 3. 스트리밍 파이프라인 재구성 (TextIteratorStreamer 사용)
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    timeout=10.0,  # 스트리밍 대기 시간
    skip_special_tokens=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.5,
    streamer=streamer,
    # generate_kwargs={"streamer": streamer}  # 스트리머 연결
)

# 4. 상태 정의 (스트리밍 상태 추가)
class AgentState(TypedDict):
    input: str
    response: str
    partial_response: str
    is_streaming: bool  # 스트리밍 진행 상태 플래그


def validate_response(state: AgentState): 
    print(f"\n\n[검증 단계] 현재 응답: {state['response']}")  # 디버깅용 출력
    if "다시" in state["response"]:
        return {"input": "더 자세한 설명을 부탁드립니다."}
    return state

# 5. 제네레이터 노드 재구성 (pipe 사용 버전)
def generate_response(state: AgentState):
    enhanced_prompt = f"[INST] {state['input']} [/INST]"
    
    # 파이프라인 스레드 시작
    generation_thread = threading.Thread(
        target=pipe,
        args=(enhanced_prompt,),  # 파이프라인에 직접 입력 전달
        kwargs={'num_return_sequences': 1}  # 기본 생성 매개변수
    )
    generation_thread.start()
    
    # 스트리밍 처리
    print(f"\n[스트리밍 단계] 현재 응답: ")  # 디버깅용 출력
    partial_response = ""
    for token in streamer:
        partial_response += token
        print(token, end="", flush=True)
        # yield {"partial_response": token, "response": partial_response, "is_streaming": True}
    
    generation_thread.join()
    return {"response": partial_response, "is_streaming": False}

# 7. 그래프 재구성 (검증 단계 복원)
workflow = StateGraph(AgentState)
workflow.add_node("generator", generate_response)
workflow.add_node("validator", validate_response)  # 검증 단계 복원
workflow.add_node("END", lambda state: state)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "validator")  # 검증 단계 연결
workflow.add_conditional_edges(
    "validator",
    lambda x: "재시도" if "다시" in x["response"] else "종료",  # 조건 수정
    {"종료": "END", "재시도": "generator"}
)

# 8. 실행
print("질문 처리 시작...\n")
app = workflow.compile()

user_prompt = input("물어보세요 : ")

# 스트리밍 실행을 위해 제네레이터로 처리
for state in app.stream({"input": user_prompt}):
    if "END" in state:
        print("\n\n=== 최종 응답 ===")
        print(state["END"]["response"])
        break