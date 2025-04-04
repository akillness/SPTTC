import os
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Export DeepSeek-R1-Distill model to ONNX format")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./deepseek-r1-finetuned",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./onnx_model",
        help="Directory to save ONNX model"
    )
    parser.add_argument(
        "--sequence_length", 
        type=int, 
        default=512,
        help="Maximum sequence length"
    )
    return parser.parse_args()

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
        # 기본 모델 로드 (CPU 모드)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            device_map="cpu",
            trust_remote_code=True
        )
        # LoRA 어댑터 적용 및 병합
        model = PeftModel.from_pretrained(base_model, model_path)
        # LoRA 가중치를 기본 모델에 병합
        model = model.merge_and_unload()
    except:
        # 전체 모델이 저장된 경우
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            trust_remote_code=True
        )
    
    # CPU 모드로 변환 및 평가 모드 설정
    model = model.to("cpu").eval()
    
    return model, tokenizer

def export_encoder_to_onnx(model, tokenizer, output_dir, max_sequence_length=512):
    """
    모델을 ONNX 형식으로 내보냅니다.
    """
    print(f"Exporting model to ONNX format...")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 더미 입력 생성 (배치 크기 1, 시퀀스 길이 max_sequence_length)
    dummy_input = "안녕하세요, 반갑습니다."
    inputs = tokenizer(dummy_input, return_tensors="pt")
    
    # ONNX 내보내기 설정
    output_path = os.path.join(output_dir, "model.onnx")
    
    # 동적 축 설정
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    }
    
    # 입력 이름 설정
    input_names = ["input_ids", "attention_mask"]
    output_names = ["output"]

    with torch.no_grad():
        # 모델 forward 함수 래퍼
        def model_forward(input_ids, attention_mask=None):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

        # ONNX 내보내기
        torch.onnx.export(
            model_forward,
            (inputs["input_ids"], inputs.get("attention_mask", None)),
            output_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    # 토크나이저 저장
    tokenizer.save_pretrained(output_dir)
    
    # 설정 파일 저장
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"max_sequence_length: {max_sequence_length}\n")
    
    print(f"Model exported to {output_path}")
    print(f"Tokenizer saved to {output_dir}")
    
    return output_path

def verify_onnx_model(onnx_model_path, tokenizer, sample_text="안녕하세요, 오늘 날씨가 좋네요."):
    """
    ONNX 모델을 검증합니다.
    """
    try:
        import onnxruntime as ort
        
        print(f"Verifying ONNX model with sample text: '{sample_text}'")
        
        # ONNX Runtime 세션 생성
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_model_path, options, providers=["CPUExecutionProvider"])
        
        # 입력 토큰화
        inputs = tokenizer(sample_text, return_tensors="pt")
        input_dict = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).numpy()
        }
        
        # ONNX 모델로 추론
        outputs = session.run(None, input_dict)
        logits = outputs[0]
        
        # 결과 확인
        predicted_ids = np.argmax(logits, axis=-1)
        predicted_text = tokenizer.decode(predicted_ids[0])
        
        print(f"ONNX Model verification successful!")
        print(f"Sample output: {predicted_text}")
        
        return True
    except ImportError:
        print("onnxruntime package not found. Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")
        return False

def main():
    args = parse_args()
    
    # 모델 로드
    model, tokenizer = load_model(args.model_path)
    
    # ONNX 모델로 내보내기
    onnx_path = export_encoder_to_onnx(
        model, 
        tokenizer, 
        args.output_dir, 
        args.sequence_length
    )
    
    # ONNX 모델 검증
    try:
        verify_onnx_model(onnx_path, tokenizer)
    except Exception as e:
        print(f"Warning: Could not verify ONNX model: {e}")
    
    print("\n=== ONNX 모델 사용법 ===")
    print("1. onnxruntime 패키지 설치:")
    print("   pip install onnxruntime")
    print("2. Python 코드에서 다음과 같이 사용:")
    print("""
import onnxruntime as ort
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("./onnx_model")

# ONNX 세션 생성
session = ort.InferenceSession("./onnx_model/model.onnx", providers=["CPUExecutionProvider"])

# 입력 생성
dialogue_type = "인터뷰"  # "알리바이", "인터뷰", "가쉽" 중 선택
persona = "20대 남성, 대학생, 취미는 게임과 독서"
question = "요즘 어떤 게임을 하고 있어?"

# 입력 형식 구성
input_text = f"<dialogue_type>\\n{dialogue_type}\\n</dialogue_type>\\n\\n<persona>\\n{persona}\\n</persona>\\n\\n<human>\\n{question}\\n</human>\\n\\n<assistant>\\n"

# 토큰화
inputs = tokenizer(input_text, return_tensors="pt")

# 입력 딕셔너리 생성
input_dict = {
    "input_ids": inputs["input_ids"].numpy(),
    "attention_mask": inputs["attention_mask"].numpy()
}

# 추론 실행
outputs = session.run(None, input_dict)
logits = outputs[0]

# 다음 토큰 예측 (간단한 예시)
import numpy as np
next_token_id = np.argmax(logits[0, -1, :])
next_token = tokenizer.decode([next_token_id])
print(f"다음 예측 토큰: {next_token}")
    """)

if __name__ == "__main__":
    main() 