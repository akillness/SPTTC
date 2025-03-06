from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",cache_dir='translate/')
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M",cache_dir='translate/')

# 입력 텍스트에 페르소나 힌트 추가
input_text = "[Formal], how are you?" # [Formal], [Casual], [Slang], [Business] 등 힌트

inputs = tokenizer(input_text, return_tensors="pt")
translated_tokens = model.generate(
        **inputs,        
        # num_beams=5,           # 더 정확한 후보 탐색
        # temperature=0.7,       # 보수적 어조
        # repetition_penalty=1.2, # 반복 단어 감소
        max_length=50,
        # 언어 코드 ID 변환
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("kor_Hang"),  # 한국어 타겟 설정, deu_Latn, eng_Latn
    )

# 번역된 텍스트 디코딩
translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print(translated_text)  # "안녕하세요, 잘 지내시나요?"

