# DeepSeek-R1-Distill-Qwen-1.5B 모델 파인튜닝

이 저장소는 DeepSeek-R1-Distill-Qwen-1.5B 모델을 파인튜닝하기 위한 코드를 포함하고 있습니다. LiarHeart 데이터셋의 페르소나 데이터를 사용하여 모델을 학습합니다.

## 필요 사항

필요한 라이브러리를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

## 데이터 구조

데이터는 `LiarHeart_dataset` 디렉토리 내 Excel 파일로 제공되며, 다음 열이 포함되어 있어야 합니다:
- `페르소나`: 캐릭터의 페르소나 정보
- `질문`: 사용자의 질문
- `답변`: 페르소나 기반 답변

각 엑셀 파일은 다음 시트를 포함할 수 있습니다:
- `알리바이_대화`: 알리바이 관련 대화 데이터
- `인터뷰_대화`: 인터뷰 관련 대화 데이터
- `가쉽_대화`: 가십 관련 대화 데이터

모든 시트는 같은 열 구조(`페르소나`, `질문`, `답변`)를 가져야 합니다.

## 모델 학습

모델을 학습하려면 다음 명령어를 실행하세요:

```bash
python train_deepseek.py
```

학습된 모델은 `deepseek-r1-finetuned` 디렉토리에 저장됩니다.

## TensorBoard로 학습 모니터링

학습 진행 상황을 실시간으로 모니터링하기 위해 TensorBoard가 구성되어 있습니다. 다음 단계를 따라 사용할 수 있습니다:

1. 학습 스크립트를 실행하세요:
```bash
python train_deepseek.py
```

2. 별도의 터미널 창에서 TensorBoard 서버를 시작하세요:
```bash
tensorboard --logdir=./deepseek-r1-finetuned/tensorboard_logs
```

3. 웹 브라우저에서 다음 주소로 접속하세요:
```
http://localhost:6006/
```

TensorBoard에서는 다음과 같은 정보를 확인할 수 있습니다:
- 훈련 및 검증 손실(train/eval loss)
- 학습률(learning rate) 변화
- 그래디언트 노름(gradient norm)
- 모델 파라미터 통계

## 학습 설정

`train_deepseek.py` 파일에서 다음 설정을 조정할 수 있습니다:

- `BATCH_SIZE`: 배치 크기
- `GRADIENT_ACCUMULATION_STEPS`: 그래디언트 누적 단계
- `LEARNING_RATE`: 학습률
- `NUM_EPOCHS`: 에폭 수
- `MAX_LENGTH`: 최대 시퀀스 길이
- `LORA_R`: LoRA 랭크
- `LORA_ALPHA`: LoRA 알파
- `LORA_DROPOUT`: LoRA 드롭아웃 비율
- `SHEET_NAMES`: 처리할 엑셀 시트 이름 목록

## ONNX 모델 내보내기

학습된 모델을 CPU에서 빠른 추론을 위해 ONNX 형식으로 내보낼 수 있습니다. 다음 명령어를 실행하세요:

```bash
python export_onnx.py --model_path ./deepseek-r1-finetuned --output_dir ./onnx_model
```

매개변수 옵션:
- `--model_path`: 학습된 모델 경로 (기본값: "./deepseek-r1-finetuned")
- `--output_dir`: ONNX 모델 저장 디렉토리 (기본값: "./onnx_model")
- `--sequence_length`: 최대 시퀀스 길이 (기본값: 512)

내보내기 스크립트는 다음 작업을 수행합니다:
1. 학습된 모델 및 토크나이저 로드
2. (LoRA를 사용한 경우) 어댑터를 기본 모델에 병합
3. ONNX 형식으로 모델 내보내기
4. 모델 검증 및 사용 예제 출력

## 학습된 모델 사용

### PyTorch 모델 사용

학습된 PyTorch 모델을 사용하려면 다음 명령어를 실행하세요:

```bash
python inference.py
```

### ONNX 모델 사용

ONNX로 내보낸 모델을 사용하려면 다음 명령어를 실행하세요:

```bash
python inference_onnx.py --model_path ./onnx_model
```

매개변수 옵션:
- `--model_path`: ONNX 모델 경로 (기본값: "./onnx_model")
- `--max_new_tokens`: 생성할 최대 토큰 수 (기본값: 256)
- `--temperature`: 샘플링 온도 (기본값: 0.7)
- `--top_p`: 상위 p 샘플링 값 (기본값: 0.95)

ONNX 모델은 CPU에서 최적화된 추론을 제공하여 일반적으로 더 빠른 응답 시간을 제공합니다.

대화형 인터페이스를 통해 모델과 대화할 수 있습니다. 실행 시 다음 과정을 따릅니다:

1. 대화 유형 선택 (알리바이, 인터뷰, 가쉽)
2. 페르소나 정보 입력
3. 질문 입력 및 응답 받기

대화 중에 `/유형 [유형명]` 명령을 사용하여 대화 유형을 변경할 수 있습니다:
- `/유형 알리바이`: 알리바이 대화 모드로 전환
- `/유형 인터뷰`: 인터뷰 대화 모드로 전환
- `/유형 가쉽`: 가쉽 대화 모드로 전환

## 코드에서 직접 모델 사용하기

### PyTorch 모델

학습된 PyTorch 모델을 코드에서 직접 사용하려면 다음 예제를 참조하세요:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 모델 경로
MODEL_PATH = "./deepseek-r1-finetuned"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# 추론 예시
dialogue_type = "인터뷰"  # "알리바이", "인터뷰", "가쉽" 중 선택
persona = "20대 남성, 대학생, 취미는 게임과 독서"
question = "요즘 어떤 게임을 하고 있어?"

# 입력 형식 구성
input_text = f"<dialogue_type>\n{dialogue_type}\n</dialogue_type>\n\n<persona>\n{persona}\n</persona>\n\n<human>\n{question}\n</human>\n\n<assistant>\n"

# 토큰화 및 추론
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_length=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95
)

# 결과 출력
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### ONNX 모델

ONNX 모델을 코드에서 직접 사용하려면 다음 예제를 참조하세요:

```python
import onnxruntime as ort
import numpy as np
import torch
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("./onnx_model")

# ONNX 세션 생성
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("./onnx_model/model.onnx", options, providers=["CPUExecutionProvider"])

# 입력 생성
dialogue_type = "인터뷰"  # "알리바이", "인터뷰", "가쉽" 중 선택
persona = "20대 남성, 대학생, 취미는 게임과 독서"
question = "요즘 어떤 게임을 하고 있어?"

# 입력 형식 구성
input_text = f"<dialogue_type>\n{dialogue_type}\n</dialogue_type>\n\n<persona>\n{persona}\n</persona>\n\n<human>\n{question}\n</human>\n\n<assistant>\n"

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

# 다음 토큰 예측 및 자동 생성 구현
# (실제 응용에서는 generate_with_onnx 함수 참조) 