import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model(model_path):
    """
    학습된 모델과 토크나이저를 로드합니다.
    """
    print(f"Loading model from {model_path}...")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # LoRA 어댑터가 저장된 모델인 경우
    try:
        # PeftModel 설정 로드
        config = PeftConfig.from_pretrained(model_path)
        # 기본 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # LoRA 어댑터 적용
        model = PeftModel.from_pretrained(base_model, model_path)
    except:
        # 전체 모델이 저장된 경우
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    return model, tokenizer

def generate_response(model, tokenizer, persona, question, 
                     max_length=512, temperature=0.7, top_p=0.95):
    """
    입력된 페르소나와 질문에 대한 응답을 생성합니다.
    """
    # 입력 형식 구성
    input_text = f"<persona>\n{persona}\n</persona>\n\n<human>\n{question}\n</human>\n\n<assistant>\n"
    
    # 토큰화
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 응답 생성
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # 응답 디코딩
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 응답 부분만 추출
    assistant_part = full_response.split("<assistant>")[-1].strip()
    
    return assistant_part

def interactive_chat():
    """
    대화형 채팅 인터페이스를 제공합니다.
    """
    model_path = "./deepseek-r1-finetuned"
    
    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print("대신 원본 모델을 로드합니다.")
        model, tokenizer = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    print("\n====== DeepSeek-R1 페르소나 채팅 ======\n")
    
    # 페르소나 설정
    print("페르소나를 입력하세요 (비워두면 기본값 사용):")
    persona = input("> ").strip()
    if not persona:
        persona = "20대 남성, 대학생, 취미는 게임과 독서, 성격은 친절하고 차분함"
        print(f"기본 페르소나를 사용합니다: {persona}\n")
    
    print("\n대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    while True:
        # 질문 입력
        question = input("사용자: ").strip()
        if question.lower() in ["quit", "exit"]:
            break
        
        # 응답 생성
        try:
            response = generate_response(model, tokenizer, persona, question)
            print(f"어시스턴트: {response}\n")
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    interactive_chat() 