# 복잡한 워크플로우
# 장점 : 상태 관리, 분기 처리

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnablePassthrough

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from typing import TypedDict, Any
import threading
from transformers import TextIteratorStreamer
import os

# 1. 모델 경로 지정
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
cache_dir = 'smolagent/model_cache'

# 2. 토크나이저 & 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    cache_dir=cache_dir,
    torch_dtype=torch.float32,  # float32로 설정
    low_cpu_mem_usage=True,  # 낮은 CPU 메모리 사용
    pad_token_id=tokenizer.eos_token_id,  # 명시적으로 pad token 설정
)

# pad token 설정
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model.generation_config.pad_token_id = model.config.eos_token_id

# 3. 스트리밍 파이프라인 재구성
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

generation_config = {
    "do_sample": True,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
    **generation_config
)

# 4. 상태 정의
class AgentState(TypedDict):
    input: str
    response: str
    partial_response: str
    is_streaming: bool

def validate_response(state: AgentState): 
    print(f"\n\n[검증 단계] 현재 응답: {state['response']}")
    if len(state["response"].strip()) < 10:
        return {"input": "더 자세한 설명을 부탁드립니다."}
    return state

# 5. 제네레이터 노드 구성
def generate_response(state: AgentState):
    system_prompt = """You are a helpful bilingual (Korean/English) AI programming assistant. 
Please provide clear and helpful responses. If the question is Korean, respond in Korean. For programming questions, include working code examples."""
    
    prompt = f"{system_prompt}\n\nQuestion: {state['input']}\nAnswer:"
    
    print("\n[생성 단계] 응답 생성 중...")
    
    # 파이프라인 스레드 시작
    generation_thread = threading.Thread(
        target=pipe,
        args=(prompt,),
        kwargs={'num_return_sequences': 1}
    )
    generation_thread.start()
    
    # 스트리밍 처리
    print(f"\n[스트리밍 단계] 현재 응답: ")
    partial_response = ""
    for token in streamer:
        partial_response += token
        print(token, end="", flush=True)
    
    return {"response": partial_response}

# 6. 그래프 구성
workflow = StateGraph(AgentState)
workflow.add_node("generator", generate_response)
workflow.add_node("validator", validate_response)
workflow.add_node("END", lambda state: state)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "validator")
workflow.add_conditional_edges(
    "validator",
    lambda x: "재시도" if len(x["response"].strip()) < 10 else "종료",
    {"종료": "END", "재시도": "generator"}
)

# 7. 실행 함수
def run_agent():
    print("DeepSeek Agent 시작...\n")
    app = workflow.compile()

    while True:
        try:
            user_prompt = input("\n질문하세요 (종료하려면 'q' 입력): ")
            if user_prompt.lower() == 'q':
                break

            for state in app.stream({"input": user_prompt}):
                if "END" in state:
                    print("\n\n=== 최종 응답 ===")
                    print(state["END"]["response"])
                    break
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류가 발생했습니다: {str(e)}")
            continue

if __name__ == "__main__":
    run_agent() 