import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, CLIPVisionModel
import warnings
warnings.filterwarnings("ignore")

class CustomEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # CLIP 비전 인코더만 초기화 (text 모델 제외)
        self.image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir='./')
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir='./').to(self.device)
        
        # DeepSeek 디코더 초기화 (범용 모델로 변경)
        try:
            # 일반 언어 모델 사용 시도 (더 자연스러운 이미지 설명 생성)
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./')
            self.decoder = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./').to(self.device)
            print(f"사용 중인 언어 모델: {model_name}")
        except Exception as e:
            # 범용 모델 로드 실패 시 코드 모델로 폴백
            print(f"범용 모델 로드 실패: {str(e)}, 코드 모델로 대체합니다.")
            model_name = "deepseek-ai/deepseek-coder-1.3b-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./')
            self.decoder = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./').to(self.device)
            print(f"사용 중인 언어 모델: {model_name}")
            
        # 토크나이저 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 이미지 특징 → 텍스트 임베딩 프로젝션 레이어
        self.projection = nn.Linear(768, self.decoder.config.hidden_size).to(self.device)  # CLIP vision은 768 차원

    def encode_image(self, image_path):
        """이미지 특징 추출 및 프로젝션"""
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        
        # CLIP 비전 인코더로 이미지 특징 추출
        vision_outputs = self.vision_encoder(**inputs)
        image_features = vision_outputs.pooler_output  # [1, 768] 풀링된 출력 사용
        
        return self.projection(image_features)  # [1, decoder_hidden_size]

    def generate_caption(self, image_features, prompt="이 이미지에 대해 자세히 설명해주세요:"):
        """이미지 특징과 프롬프트 결합 후 생성"""
        try:
            # 프롬프트 토크나이징
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True,
                max_length=32,
                add_special_tokens=True
            ).to(self.device)
            
            # 이미지 특징 + 텍스트 임베딩 결합 복원
            text_embeds = self.decoder.get_input_embeddings()(inputs.input_ids)
            image_embeds = image_features.unsqueeze(1)
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            
            # 어텐션 마스크 조정 (차원 일치 보장)
            attention_mask = torch.cat([
                torch.ones(1, 1).to(self.device),  # 이미지 토큰 마스크
                inputs.attention_mask
            ], dim=1)
            
            # 생성 실행
            outputs = self.decoder.generate(
                inputs_embeds=combined_embeds,  # 이미지 + 텍스트 결합 임베딩 사용
                attention_mask=attention_mask,  # 결합된 어텐션 마스크 사용
                max_new_tokens=150,
                num_beams=3,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2  # 반복 방지 패널티 추가
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 제거하고 생성된 텍스트만 반환
            if prompt in generated_text:
                generated_text = generated_text.split(prompt, 1)[1]
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return None

def main():
    try:
        model = CustomEncoderDecoder()
        model.eval()
        
        image_path = "test.jpeg"
        print(f"\n이미지 파일: {image_path}")
        with torch.no_grad():
            image_features = model.encode_image(image_path)
            caption = model.generate_caption(image_features)
            print(f"\n생성된 캡션: {caption}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()