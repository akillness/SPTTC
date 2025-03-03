from transformers import (
    ViTModel, 
    AutoModelForCausalLM, 
    ViTImageProcessor, 
    AutoTokenizer,
    EncoderDecoderModel,
    PretrainedConfig
)
import torch
from PIL import Image

# 1. 인코더(ViT)와 디코더(DeepSeek) 로드
encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k",cache_dir='./')
decoder = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",cache_dir='./')

# 2. Encoder-Decoder 아키텍처 결합 --> 커스텀 모델 설정
class VisionToTextModel(EncoderDecoderModel):
    def __init__(self, encoder, decoder):
        config = PretrainedConfig.from_pretrained(decoder.config.name_or_path)
        super().__init__(encoder=encoder, decoder=decoder, config=config)
        
    def forward(self, pixel_values, decoder_input_ids, **kwargs):
        encoder_outputs = self.encoder(pixel_values=pixel_values)
        return self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            **kwargs
        )

model = VisionToTextModel(encoder, decoder)

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
    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=100,
        num_beams=5,
        early_stopping=True
    )

# 결과 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Caption: {generated_text}")