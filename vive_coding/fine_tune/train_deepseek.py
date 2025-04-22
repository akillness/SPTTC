import os
import glob
import pandas as pd
import torch
import unicodedata
import torch.multiprocessing as mp
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
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers.integrations import TensorBoardCallback
from huggingface_hub import login

# Set multiprocessing start method to 'spawn'
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)


# 데이터 경로 설정
DATA_DIR = os.path.abspath("LiarHeart_dataset")
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "deepseek-r1-finetuned"
TB_LOG_DIR = os.path.join(OUTPUT_DIR, "tensorboard_logs")

# LoRA 설정 (메모리 효율성 중심)
LORA_R = 8                # 기존 16 → 낮은 랭크로 메모리 절약
LORA_ALPHA = 16           # alpha = 2*R 권장
LORA_DROPOUT = 0.15       # 약간의 정규화 강화
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"  # MLP 레이어 추가
]

# 학습 설정 (L4 최적화)
BATCH_SIZE = 2                     # 4 → 2로 감소
GRADIENT_ACCUMULATION_STEPS = 16   # 8 → 16으로 증가 (유효 배치 크기 유지)
LEARNING_RATE = 3e-4               # 기존 5e-4 → 낮은 LR로 안정성 확보
NUM_EPOCHS = 1                     # 변동 없음 (소형 모델 특성)
MAX_LENGTH = 512                   # 변동 없음 (VRAM 한계)
WARMUP_RATIO = 0.05                # 기존 0.1 → 빠른 워밍업
WEIGHT_DECAY = 0.01                # 기존 0.05 → 과적합 방지 조정

# 처리할 시트 목록
SHEET_NAMES = ["알리바이_대화", "인터뷰_대화", "가쉽_대화"]

# 커스텀 콜백 클래스
class CustomCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  # 10 스텝마다 진행상황 출력
            print(f"Step {state.global_step}/{state.max_steps} - Loss: {state.log_history[-1]['loss']:.4f}")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nEpoch {state.epoch} completed\n")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n===== Evaluation Results at Step {state.global_step} =====")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            print("=" * 50)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            print(f"[Progress] {state.global_step/state.max_steps*100:.1f}% 완료")
            print(f"[메모리 사용량] GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# 데이터 로딩 및 준비 함수
def load_and_prepare_data():
    # Excel 파일 찾기
    search_prefix = "페르소나 데이터_"
    search_suffix = ".xlsx"
    normalized_prefix = unicodedata.normalize('NFC', search_prefix)

    print(f"Searching for files in {DATA_DIR}")
    EXCEL_FILES = []
    for filename in os.listdir(DATA_DIR):
        normalized_filename = unicodedata.normalize('NFC', filename)
        if normalized_filename.startswith(normalized_prefix) and normalized_filename.endswith(search_suffix) and not normalized_filename.startswith("~$"):
            EXCEL_FILES.append(os.path.join(DATA_DIR, filename))
    print(f"Found Excel files: {EXCEL_FILES}")
    
    dialogue_datas = []    
    for excel_file in tqdm(EXCEL_FILES, desc="Loading Excel files"):
        print(f"Processing file: {excel_file}")

        persona_name = excel_file.split('_')[-1].split('.')[0]
        person_dialogue_dfs = []
        for sheet_name in SHEET_NAMES:
            try:
                print(f"  Reading sheet: {sheet_name}")
                # Read Excel with string type for all columns
                df = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    dtype=str  # Force string type during reading
                )
                print(f"  Columns in sheet {sheet_name}: {df.columns.tolist()}")
                
                # Select only the required columns
                required_columns = ['사람 대사', '챗봇 대사', '감정']
                if all(col in df.columns for col in required_columns):
                    df = df[required_columns].copy()
                    # Clean the data
                    df = df.fillna('')  # Replace NaN with empty string
                    df['이름'] = persona_name  # Add persona name
                    person_dialogue_dfs.append(df)
                else:
                    print(f"  Warning: Required columns not found in sheet {sheet_name}")
                    print(f"  Available columns: {df.columns.tolist()}")
                    print(f"  Required columns: {required_columns}")
            except Exception as e:
                print(f"  Error processing {excel_file}, sheet {sheet_name}: {e}")

        if person_dialogue_dfs:
            dialoguse = pd.concat(person_dialogue_dfs, ignore_index=True)
            dialogue_datas.append(dialoguse)

    if not dialogue_datas:
        raise ValueError("No data was loaded from the Excel files")

    persona_datas = pd.concat(dialogue_datas, ignore_index=True)
    
    # Clean and prepare data
    persona_datas = persona_datas.fillna('')  # Replace any remaining NaN
    persona_datas = persona_datas.astype(str)  # Ensure string type
    
    # Rename columns to match expected format
    column_mapping = {
        '사람 대사': 'Q',
        '챗봇 대사': 'A',
        '감정': 'E',
        '이름': 'N'
    }
    persona_datas = persona_datas.rename(columns=column_mapping)
    
    # Handle emotion field
    persona_datas['E'] = persona_datas['E'].replace({'': '감정없음', 'nan': '감정없음', 'None': '감정없음'})
    
    # Remove any rows with empty essential fields
    persona_datas = persona_datas[
        (persona_datas['Q'].str.strip() != '') & 
        (persona_datas['A'].str.strip() != '')
    ].reset_index(drop=True)
    
    print(f"Total examples loaded: {len(persona_datas)}")
    print(f"Final columns: {persona_datas.columns.tolist()}")

    # Format the text for training
    texts = []
    for _, row in persona_datas.iterrows():
        formatted_text = f"Human: {row['Q'].strip()}\nAssistant: {row['A'].strip()}\nEmotion: {row['E'].strip()}\nName: {row['N'].strip()}"
        texts.append(formatted_text)

    # 학습:검증 데이터 분리 (9:1)
    train_size = int(len(texts) * 0.9)
    
    # Create datasets directly from lists
    try:
        train_dataset = Dataset.from_dict({"text": texts[:train_size]})
        eval_dataset = Dataset.from_dict({"text": texts[train_size:]})
        
        # Verify the datasets
        print("Train dataset size:", len(train_dataset))
        print("Eval dataset size:", len(eval_dataset))
        
        # Verify data type
        print("Sample text from train dataset:", train_dataset[0]['text'][:100])
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        print("Error creating datasets:", str(e))
        raise

# 모델과 토크나이저 준비 함수
def prepare_model_and_tokenizer():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, cache_dir=DATA_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        cache_dir=DATA_DIR
    )

    # Enable gradient computation
    model.config.use_cache = False  # Disable cache for training
    
    # LoRA 설정 적용
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )

    print("Preparing PEFT model...")
    model = get_peft_model(model, peft_config)
    
    # Enable training for all LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            print(f"Parameter {name} requires_grad: {param.requires_grad}")
        else:
            param.requires_grad = False

    # Convert trainable parameters to float32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    return model, tokenizer

# 학습 설정
class GPUDataCollator:
    def __init__(self, base_collator, device):
        self.base_collator = base_collator
        self.device = device

    def __call__(self, examples):
        batch = self.base_collator(examples)
        # Move batch to GPU
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

def main():
    # 데이터 준비
    train_dataset, eval_dataset = load_and_prepare_data()

    # 모델 및 토크나이저 준비
    model, tokenizer = prepare_model_and_tokenizer()

    # Get model's device
    device = next(model.parameters()).device
    print(f"\nModel is on device: {device}")

    # 데이터 토크나이징
    print("Tokenizing datasets...")
    
    def tokenize_function(examples):
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="longest",
            return_tensors="pt"
        )

    print("Tokenizing training dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training dataset"
    )

    print("Tokenizing validation dataset...")
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation dataset"
    )

    # 학습 설정
    base_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Wrap the data collator with GPU support
    data_collator = GPUDataCollator(base_data_collator, device)

    os.makedirs(TB_LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,  # CPU 코어 수에 맞게 증가
        dataloader_prefetch_factor=2,  # 추가
        dataloader_persistent_workers=True,  # 추가
        remove_unused_columns=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        group_by_length=True,
        prediction_loss_only=True,
        label_names=["labels"],
    )

    # Trainer 초기화 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[CustomCallback(), TensorBoardCallback()],
    )

    # Move model to device before training
    model.to(device)

    # 학습 시작
    print("Starting training...")
    try:
        # Verify model state before training
        print("\nModel state before training:")
        print(f"Training mode: {model.training}")
        print(f"Device: {device}")
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: requires_grad={param.requires_grad}, dtype={param.dtype}, device={param.device}")

        # Verify data processing
        print("\nVerifying data processing:")
        sample_batch = data_collator([tokenized_train[0]])
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")

        # Start training
        trainer.train()
        
        # Run evaluation
        print("Running evaluation...")
        eval_results = trainer.evaluate()
        print("\n===== Evaluation Results =====")
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nDebug information:")
        print(f"Model state: {model.training}")
        print(f"Model device: {device}")
        print("Sample data batch:")
        sample_batch = data_collator([tokenized_train[0]])
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")
                # Move tensor to CPU for inspection
                v = v.cpu()
                print(f"First few values: {v.flatten()[:5]}")
        raise

    # 모델 저장
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete. Model saved to {OUTPUT_DIR}")
    print(f"TensorBoard logs saved to {TB_LOG_DIR}")

if __name__ == "__main__":
    main() 