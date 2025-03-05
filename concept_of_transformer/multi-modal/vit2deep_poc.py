from transformers import (
    ViTModel, 
    AutoModelForCausalLM, 
    ViTImageProcessor, 
    AutoTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    PretrainedConfig
)

import torch
import torch.nn as nn

from PIL import Image
'''
Zero-Shot 이미지 캡셔닝의 기본 프레임워크
'''
# # 0. 차원 불일치 해결 (CustomViTEncoder Layer 추가)
# class CustomViTEncoder(ViTModel):
#     def __init__(self, eoncoder_config,decoder_config):
#         super().__init__(eoncoder_config)
#         # 커스텀 레이어 추가 (예: Linear Layer)
#         self.custom_layer = nn.Linear(eoncoder_config.hidden_size, decoder_config.hidden_size)
    
#     # 4. Forward 함수 오버라이드
#     def forward(self, pixel_values):
#         # 기존 ViT forward
#         outputs = super().forward(pixel_values)
#         last_hidden_states = outputs.last_hidden_state
        
#         # 커스텀 조정
#         adjusted_states = self.custom_layer(last_hidden_states)
#         return adjusted_states
    

# 0. Encoder-Decoder 직접 연결
class CustomEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, pixel_values, input_ids):
        # 인코더 출력
        encoder_outputs = self.encoder(pixel_values)
        
        # 디코더에 전달 (커스텀 로직 추가)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_outputs
        )
        return decoder_outputs

# from transformers import AutoModelForCausalLM, AutoConfig
# import torch.nn as nn

class CustomDeepSeekDecoder(nn.Module):
    def __init__(self, decoder_config, encoder_hidden_size):
        super().__init__()
        # 원본 DeepSeek 디코더 로드
        self.decoder = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            cache_dir="./",
            config=decoder_config
        )
        
        # 크로스 어텐션 레이어 추가
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_config.hidden_size,
            num_heads=decoder_config.num_attention_heads,
            kdim=encoder_hidden_size,
            vdim=encoder_hidden_size
        )
        self.layer_norm = nn.LayerNorm(decoder_config.hidden_size)

    def forward(self, input_ids, encoder_hidden_states):
        # 기존 디코더 출력
        outputs = self.decoder(input_ids=input_ids)
        last_hidden_states = outputs.logits
        
        # 크로스 어텐션 적용
        cross_attn_output, _ = self.cross_attention(
            query=last_hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states
        )
        adjusted_output = self.layer_norm(last_hidden_states + cross_attn_output)
        return adjusted_output

# 사용 예시
# decoder_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# custom_decoder = CustomDeepSeekDecoder(decoder_config, encoder_hidden_size=768)

# 1. Encoder와 Decoder를 별도로 로드 (cache_dir 지정)
encoder = ViTModel.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    cache_dir="./"  # 캐시 디렉토리 명시
)

decoder = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir="./"  # 캐시 디렉토리 명시
)

# 2. Encoder-Decoder custom encoder 생성
# custom_vitencoder = CustomViTEncoder(encoder.config,decoder.config)

# # 3. Encoder-Decoder Config 생성
# config = EncoderDecoderConfig.from_encoder_decoder_configs(
#     encoder_config=custom_vitencoder.config,
#     decoder_config=decoder.config
# )

# 3. EncoderDecoderModel에 직접 주입
model = CustomEncoderDecoder(
    encoder=encoder,
    decoder=decoder,
)

# # 4. Encoder-Decoder 모델 초기화 (cache_dir 지정)
# model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#     encoder_pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
#     decoder_pretrained_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     encoder_kwargs={"cache_dir": "./"},  # encoder 캐시 경로
#     decoder_kwargs={"cache_dir": "./"},  # decoder 캐시 경로
#     config=config
#)



'''
# 학습시 decoder만 학습하는 방법
for param in model.encoder.parameters():
    param.requires_grad = False
'''

''' 전처리 파이프라인 설정 '''
# 이미지 프로세서 (ViT용)
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k",cache_dir='./')

# 텍스트 토크나이저 (DeepSeek용)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",cache_dir='./')
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정

def process_data(image_path):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    
    # 디코더 입력 생성 (시작 토큰 추가)
    decoder_inputs = tokenizer(
        tokenizer.bos_token,  # <bos> 토큰으로 초기화
        return_tensors="pt",
        add_special_tokens=False
    )
    return pixel_values, decoder_inputs.input_ids

# 이미지 경로 설정
image_path = "test.jpeg"

# 데이터 전처리
pixel_values, decoder_input_ids = process_data(image_path)

# 모델 추론
with torch.no_grad():
    # outputs = model.generate(
    #     pixel_values=pixel_values,
    #     decoder_input_ids=decoder_input_ids,
    #     max_length=100,
    #     num_beams=5,
    #     early_stopping=True
    # )
    
    outputs = model(pixel_values=pixel_values, input_ids=decoder_input_ids)

# 결과 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Caption: {generated_text}")