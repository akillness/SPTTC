from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

model = AutoModelForTokenClassification.from_pretrained("rv2307/electra-small-ner",cache_dir='./')
tokenizer = AutoTokenizer.from_pretrained("rv2307/electra-small-ner",cache_dir='./')

nlp = pipeline("ner",
              model=model,
              tokenizer=tokenizer,device="mps",
              aggregation_strategy="max")

# 텍스트 입력 및 처리
# text = input("NER 분석을 원하는 텍스트를 입력하세요: ")
text = "The quick brown fox jumps over the dog"
ner_results = nlp(text)

# 결과 출력
print("\n[개체명 인식 결과]")
if ner_results:
    for entity in ner_results:
        print(f"- {entity['entity_group']}: {entity['word']} (시작 위치: {entity['start']}, 끝 위치: {entity['end']})")
else:
    print("인식된 개체명이 없습니다.")