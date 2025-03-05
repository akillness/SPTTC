import torch
from PIL import Image
from transformers import ViTModel, GPT2LMHeadModel, ViTImageProcessor, GPT2Tokenizer

# 1. Adapter 및 멀티모달 모델 정의 (이전 코드와 동일)
class ImageToTextAdapter(torch.nn.Module):
    def __init__(self, image_embed_dim=768, text_embed_dim=768):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(text_embed_dim)
        )

    def forward(self, image_embeddings):
        return self.adapter(image_embeddings)

class MultimodalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir='./')
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',cache_dir='./')
        self.adapter = ImageToTextAdapter()
        
        # 가중치 고정
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.gpt.parameters():
            param.requires_grad = False

# 2. 추론 파이프라인 함수
def generate_image_caption(image_path, max_length=30):
    # 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalModel().to(device)
    model.eval()
    
    # 이미지 처리
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir='./')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',cache_dir='./')
    tokenizer.pad_token = tokenizer.eos_token
    
    
    # 이미지 로드 및 임베딩 추출
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vit_outputs = model.vit(**inputs)
    image_emb = vit_outputs.last_hidden_state.mean(dim=1)  # [1, 768]
    adapted_emb = model.adapter(image_emb).unsqueeze(1)    # [1, 1, 768]

    # GPT 입력 생성
    prompt = "This image shows"
    # 3. 토크나이징 및 attention_mask 생성
    inputs = tokenizer(
        prompt,
        return_tensors='pt',          # PyTorch 텐서 반환
        padding=True,                 # 패딩 적용
        truncation=True,              # 최대 길이 초과 시 잘라냄
        return_attention_mask=True    # attention_mask 반환 필수!
    )

    text_emb = model.gpt.transformer.wte(inputs.input_ids)        # [1, seq_len, 768]

    # 1. 이미지 토큰을 위한 attention_mask 확장 (중요!)
    image_mask = torch.ones(1, 1).to(device)  # 이미지 토큰 마스크 (값=1)
    attention_mask = torch.cat([image_mask, inputs.attention_mask], dim=1)  # [1, seq_len+1]

    # 2. 멀티모달 임베딩 결합
    combined_emb = torch.cat([adapted_emb, text_emb], dim=1)  # [1, seq_len+1, 768]

    # 3. 생성 시 확장된 attention_mask 사용
    generated_ids = model.gpt.generate(
        inputs_embeds=combined_emb,
        attention_mask=attention_mask,  # 수정된 마스크 전달
        max_length=max_length,
        num_beams=5,
        # temperature=0.1,
        early_stopping=True,
        # used_cached=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 3. 실행 예시
if __name__ == "__main__":
    image_path = "test.jpeg"  # 실제 이미지 경로로 변경
    caption = generate_image_caption(image_path)
    print(f"생성된 설명: {caption}")
 