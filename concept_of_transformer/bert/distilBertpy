import torch
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score

# 1. 데이터 준비
texts = [
    "문을 열어주세요.",
    "오늘 날씨가 좋네요.",
    "보고서를 작성해 주세요.",
    "점심 메뉴는 무엇인가요?"
]
labels = [1, 0, 1, 0]  # 1: 명령문, 0: 일반문

# 2. 데이터셋 분할 (80% train, 20% test)
dataset = Dataset.from_dict({'text': texts, 'labels': labels}).train_test_split(test_size=0.2, seed=42)

# 3. 토크나이저 로드
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',cache_dir='bert/')

# 4. 토큰화 함수 정의
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# 5. 데이터셋 토큰화
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 6. 모델 로드
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2,
    cache_dir='./'
)

# 7. 평가 지표 함수
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}

# 8. 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy='no',
    report_to='none'
)

# 9. Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)

# 10. 모델 학습
trainer.train()

# 11. 모델 평가
results = trainer.evaluate()
print(f"최종 평가 결과: {results}")