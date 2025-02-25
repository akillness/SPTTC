import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, EncoderDecoderModel

import line_profiler
# ㄴ 사용 방법 : kernprof -l -v test.py 
    
import timeit

def timeit_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        print(f"{func.__name__} 실행 시간: {execution_time}초")
        return result
    return wrapper

# 디바이스 설정 (GPU가 사용 가능하면 GPU, 아니면 CPU)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#@profile
@timeit_decorator
def load_model(model_name, cache_dir = ""):    
    # 설정 로드
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    '''
    if device == "cpu":
        # 양자화 설정 제거
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
    '''

    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    # decoder 모델 로드시 사용
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, config=config, trust_remote_code=True)
    
    return model, tokenizer

#@profile
# @timeit_decorator
def generate(model,tokenizer,input_text, device):
    # 입력을 토큰화하고 텐서로 변환
    device = torch.device(device)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 모델을 사용하여 출력 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=1000)

    # 출력된 토큰을 문자열로 변환하여 결과 출력
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
    
'''
from huggingface_hub import login

# 생성한 액세스 토큰을 입력합니다.
login(token="")
'''

def main():
    
    # 모델 이름
    # model_name = "deepseek-ai/DeepSeek-R1"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # 원하는 캐시 디렉토리 경로
    custom_cache_dir = './'

    
    model, tokenizer = load_model(model_name,cache_dir=custom_cache_dir)

    # BERT를 인코더로, GPT-2를 디코더로 사용하는 인코더-디코더 모델 로드
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")

    # 입력 문장
    input_text = "DeepSeek R1 모델의 주요 특징은 무엇인가요?"
    response = generate(model,tokenizer,input_text,device)
    
    print(response)

    '''
    from transformers import EncoderDecoderModel, BertTokenizer, GPT2Tokenizer

    # 인코더와 디코더의 토크나이저 로드
    encoder_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 인코더-디코더 모델 로드
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")

    # 입력 시퀀스
    input_text = "Hugging Face는 훌륭한 NLP 라이브러리를 제공합니다."
    input_ids = encoder_tokenizer(input_text, return_tensors="pt").input_ids

    # 인코더를 통해 입력을 인코딩하여 잠재 변수 z 획득
    encoder_outputs = model.encoder(input_ids)
    latent_z = encoder_outputs.last_hidden_state

    # 디코더 입력 준비 (예: 시작 토큰)
    decoder_input_ids = decoder_tokenizer("<|startoftext|>", return_tensors="pt").input_ids

    # 디코더를 통해 출력 생성
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    # 생성된 텍스트 디코딩
    generated_text = decoder_tokenizer.decode(outputs.logits.argmax(-1).squeeze(), skip_special_tokens=True)
    print(generated_text)
    '''

# 프로파일러 실행
if __name__ == '__main__':
    main()
    
    