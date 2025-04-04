import os
import glob
import pandas as pd
import torch
import unicodedata
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers.integrations import TensorBoardCallback



# 데이터 디렉토리 설정
DATA_DIR = os.path.abspath("./LiarHeart_dataset")  # Look for dataset in the same directory as the script
print(f"Data directory: {DATA_DIR}")
print(f"Current working directory: {os.getcwd()}")

# Excel 파일 찾기
search_prefix = "페르소나 데이터_"
search_suffix = ".xlsx"
normalized_prefix = unicodedata.normalize('NFC', search_prefix)

print(f"Searching for files starting with '{normalized_prefix}' and ending with '{search_suffix}' in {DATA_DIR}")
EXCEL_FILES = []
for filename in os.listdir(DATA_DIR):
    normalized_filename = unicodedata.normalize('NFC', filename)
    if normalized_filename.startswith(normalized_prefix) and normalized_filename.endswith(search_suffix) and not normalized_filename.startswith("~$"):
        EXCEL_FILES.append(os.path.join(DATA_DIR, filename))
print(f"Found Excel files: {EXCEL_FILES}")

# 디렉토리 내용 확인
print("\nDirectory contents:")
for item in os.listdir(DATA_DIR):
    print(f"- {item}")

# 처리할 시트 목록
SHEET_NAMES = ["알리바이_대화", "인터뷰_대화", "가쉽_대화"]

# 모델 설정
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "./deepseek-r1-finetuned"
# TensorBoard 로그 디렉토리 설정
TB_LOG_DIR = os.path.join(OUTPUT_DIR, "tensorboard_logs")

# LoRA 설정 최적화
LORA_R = 16  # 증가: 더 많은 파라미터 학습
LORA_ALPHA = 32  # 증가: 더 강한 적응력
LORA_DROPOUT = 0.1  # 증가: 더 강한 정규화
# OPT 모델에 맞는 타겟 모듈 지정
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 학습 설정 최적화
BATCH_SIZE = 2  # 메모리 가용성에 따라 조정
GRADIENT_ACCUMULATION_STEPS = 16  # 효과적인 배치 크기 유지
LEARNING_RATE = 5e-4  # 증가: 더 빠른 학습
NUM_EPOCHS = 1  # 감소: 빠른 테스트 용도
MAX_LENGTH = 512
WARMUP_RATIO = 0.1  # 증가: 더 안정적인 초기 학습
WEIGHT_DECAY = 0.05  # 증가: 더 강한 정규화

# 커스텀 콜백 클래스 추가
class LearningRateLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        optimizer = kwargs.get('optimizer', None)
        if optimizer:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                print(f"Step {state.global_step}: Learning rate = {current_lr:.6f}")
        return control
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n===== Evaluation Results at Step {state.global_step} =====")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            print("=" * 50)
        return control

def load_and_prepare_data():
    """
    pandas를 사용하여 엑셀 파일을 읽고 모델 학습을 위한 데이터셋을 준비합니다.
    여러 시트("알리바이_대화", "인터뷰_대화", "가쉽_대화")를 모두 처리합니다.
    """
    all_data = []
    # first_file_processed = False # Flag removed, no longer needed
    
    for excel_file in tqdm(EXCEL_FILES, desc="Loading Excel files"):
        if "~$" in excel_file:  # 임시 파일 제외
            continue
        
        print(f"Processing file: {excel_file}")

        # # --- Debugging: Print sheet names for the first file --- (Removed)
        # if not first_file_processed:
        #     try:
        #         xls = pd.ExcelFile(excel_file)
        #         print(f"  Available sheets in {os.path.basename(excel_file)}: {xls.sheet_names}")
        #         first_file_processed = True
        #     except Exception as e:
        #         print(f"  Error reading sheet names from {excel_file}: {e}")
        # # --- End Debugging ---

        # 각 시트 처리
        for sheet_name in SHEET_NAMES:
            # Skip "검증 질문" sheet for now as it has a different structure and purpose
            if sheet_name == "검증 질문": 
                continue
            
            try:
                print(f"  Reading sheet: {sheet_name}")
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # # --- Debugging: Print columns for the current sheet --- (Removed)
                # print(f"    Columns in '{sheet_name}': {df.columns.tolist()}")
                # # --- End Debugging ---
                
                # Use actual column names: '사람 대사' and '챗봇 대사'
                human_col = '사람 대사'
                assistant_col = '챗봇 대사'

                if human_col in df.columns and assistant_col in df.columns:
                    for _, row in df.iterrows():
                        question = row[human_col]
                        answer = row[assistant_col]
                        
                        # Ensure question and answer are strings and not NaN
                        if pd.isna(question) or pd.isna(answer):
                            continue
                            
                        question = str(question)
                        answer = str(answer)

                        # Updated formatting without persona
                        dialogue_type = ""
                        if sheet_name == "알리바이_대화": dialogue_type = "알리바이"
                        elif sheet_name == "인터뷰_대화": dialogue_type = "인터뷰"
                        elif sheet_name == "가쉽_대화": dialogue_type = "가쉽"
                        
                        if dialogue_type:
                            formatted_text = f"<dialogue_type>\n{dialogue_type}\n</dialogue_type>\n\n<human>\n{question}\n</human>\n\n<assistant>\n{answer}\n</assistant>"
                            all_data.append({"text": formatted_text})
                else:
                    print(f"  Warning: Required columns '{human_col}' or '{assistant_col}' not found in {excel_file}, sheet {sheet_name}")
                        
            except Exception as e:
                if "Worksheet named" in str(e) and "not found" in str(e):
                     print(f"  Info: Sheet '{sheet_name}' does not exist in {excel_file}")
                else:
                    print(f"  Error processing {excel_file}, sheet {sheet_name}: {e}")
    
    print(f"Total examples loaded: {len(all_data)}")
    
    # 학습:검증 데이터 분리 (9:1)
    train_size = int(len(all_data) * 0.9)
    train_data = all_data[:train_size]
    eval_data = all_data[train_size:]
    
    return Dataset.from_dict({"text": [item["text"] for item in train_data]}), \
           Dataset.from_dict({"text": [item["text"] for item in eval_data]})

def prepare_model_and_tokenizer():
    """
    모델과 토크나이저를 준비하고 LoRA 설정을 적용합니다.
    """
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, cache_dir="./")
    
    # 8비트 양자화를 사용하여 메모리 사용량 감소
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # 16비트 부동소수점 사용
        device_map="auto",
        cache_dir="./"
    )
    
    # 8비트 학습을 위한 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정 적용
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",  # 메모리 효율성 향상
        inference_mode=False,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    """
    텍스트를 토크나이징합니다.
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def train():
    """
    모델 파인튜닝을 실행합니다.
    """
    # 데이터 준비
    train_dataset, eval_dataset = load_and_prepare_data()
    
    # 모델 및 토크나이저 준비
    model, tokenizer = prepare_model_and_tokenizer()
    
    # 데이터 토크나이징
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # TensorBoard 디렉토리 생성
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_dir=TB_LOG_DIR,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # 16비트 학습 활성화
        bf16=False, # 이미 fp16을 사용하므로 비활성화
        remove_unused_columns=False,
        report_to=["tensorboard"],
        gradient_checkpointing=True,  # 그래디언트 체크포인팅 활성화
        optim="adamw_torch",  # 최적화된 옵티마이저 사용
        ddp_find_unused_parameters=False,  # 성능 향상
    )
    
    # 커스텀 콜백 생성
    lr_callback = LearningRateLoggerCallback()
    tensorboard_callback = TensorBoardCallback()
    
    # 학습 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[lr_callback, tensorboard_callback],  # 커스텀 콜백 추가
    )
    
    # 학습 진행 상황 안내
    print("\n===== TensorBoard 실행 방법 =====")
    print(f"터미널에서 다음 명령어를 실행하세요:")
    print(f"tensorboard --logdir={TB_LOG_DIR}")
    print("그런 다음 웹 브라우저에서 http://localhost:6006/ 으로 접속하세요.\n")
    
    # 초기 평가 실행
    print("Initial evaluation...")
    trainer.evaluate()
    
    # 학습 실행
    print("Starting training...")
    trainer.train()
    
    # 모델 및 토크나이저 저장
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 최종 평가
    print("Final evaluation...")
    final_metrics = trainer.evaluate()
    print("\n===== Final Evaluation Results =====")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)
    
    print(f"Training complete. Model saved to {OUTPUT_DIR}")
    print(f"TensorBoard 로그는 {TB_LOG_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    train() 