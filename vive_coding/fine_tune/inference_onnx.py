import argparse
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Optional, Tuple
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with ONNX model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./onnx_model",
        help="Path to the ONNX model directory"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.95,
        help="Top-p for sampling"
    )
    return parser.parse_args()

def initialize_model(model_path: str):
    """
    Initialize ONNX model and tokenizer
    """
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create ONNX session
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    onnx_model_path = f"{model_path}/model.onnx"
    session = ort.InferenceSession(
        onnx_model_path, 
        options, 
        providers=["CPUExecutionProvider"]
    )
    
    print("Model loaded successfully")
    return session, tokenizer

def generate_with_onnx(
    session: ort.InferenceSession,
    tokenizer,
    dialogue_type: str,
    persona: str,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> str:
    """
    Generate response using ONNX model with basic sampling capabilities
    """
    start_time = time.time()
    
    # Prepare input
    input_text = f"<dialogue_type>\n{dialogue_type}\n</dialogue_type>\n\n<persona>\n{persona}\n</persona>\n\n<human>\n{question}\n</human>\n\n<assistant>\n"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    
    # Track generated tokens
    all_token_ids = input_ids.tolist()[0]
    
    # Keep track of attention mask
    current_attention_mask = attention_mask.copy()
    
    print(f"Starting generation with {len(all_token_ids)} input tokens")
    
    # Autoregressive generation
    for _ in range(max_new_tokens):
        current_input_ids = np.array([all_token_ids], dtype=np.int64)
        current_attention_mask = np.ones((1, len(all_token_ids)), dtype=np.int64)
        
        # Create input dict
        input_dict = {
            "input_ids": current_input_ids,
            "attention_mask": current_attention_mask
        }
        
        # Run inference
        outputs = session.run(None, input_dict)
        logits = outputs[0]
        
        # Get logits for the last token
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = np.sort(next_token_logits)[::-1], np.argsort(next_token_logits)[::-1]
            cumulative_probs = np.cumsum(torch.nn.functional.softmax(torch.tensor(sorted_logits), dim=-1).numpy())
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float("Inf")
        
        # Sample from the distribution
        probs = torch.nn.functional.softmax(torch.tensor(next_token_logits), dim=-1).numpy()
        next_token_id = np.random.choice(len(probs), p=probs)
        
        # Stop if end token is predicted
        if next_token_id == tokenizer.eos_token_id:
            break
            
        # Add the predicted token to the input
        all_token_ids.append(int(next_token_id))
        
        # Print progress for longer generations
        if len(all_token_ids) % 10 == 0:
            print(f"Generated {len(all_token_ids) - len(input_ids[0])} tokens...")
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    
    # Extract assistant response
    try:
        assistant_part = generated_text.split("<assistant>")[-1].strip()
    except:
        assistant_part = generated_text
    
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    return assistant_part

def interactive_chat():
    """
    대화형 채팅 인터페이스를 제공합니다.
    """
    args = parse_args()
    
    # Initialize model
    try:
        session, tokenizer = initialize_model(args.model_path)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    print("\n====== DeepSeek-R1 ONNX 채팅 ======\n")
    
    # 대화 유형 선택
    print("대화 유형을 선택하세요 (1: 알리바이, 2: 인터뷰, 3: 가쉽):")
    dialogue_type_choice = input("> ").strip()
    if dialogue_type_choice == "1":
        dialogue_type = "알리바이"
    elif dialogue_type_choice == "2":
        dialogue_type = "인터뷰"
    elif dialogue_type_choice == "3":
        dialogue_type = "가쉽"
    else:
        dialogue_type = "알리바이"  # 기본값
        print(f"유효하지 않은 선택, 기본값 '{dialogue_type}'을 사용합니다.")
    
    print(f"\n선택한 대화 유형: {dialogue_type}")
    
    # 페르소나 설정
    print("\n페르소나를 입력하세요 (비워두면 기본값 사용):")
    persona = input("> ").strip()
    if not persona:
        persona = "20대 남성, 대학생, 취미는 게임과 독서, 성격은 친절하고 차분함"
        print(f"기본 페르소나를 사용합니다: {persona}")
    
    print("\n대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("대화 유형을 변경하려면 '/유형 알리바이', '/유형 인터뷰' 또는 '/유형 가쉽'을 입력하세요.\n")
    
    while True:
        # 질문 입력
        question = input("사용자: ").strip()
        if question.lower() in ["quit", "exit"]:
            break
        
        # 대화 유형 변경 명령 확인
        if question.startswith("/유형 "):
            new_type = question[4:].strip()
            if new_type in ["알리바이", "인터뷰", "가쉽"]:
                dialogue_type = new_type
                print(f"대화 유형이 '{dialogue_type}'(으)로 변경되었습니다.")
            else:
                print(f"지원되지 않는 대화 유형입니다. '알리바이', '인터뷰', '가쉽' 중에서 선택하세요.")
            continue
        
        # 응답 생성
        try:
            response = generate_with_onnx(
                session, 
                tokenizer, 
                dialogue_type, 
                persona, 
                question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"어시스턴트 [{dialogue_type}]: {response}\n")
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    interactive_chat() 