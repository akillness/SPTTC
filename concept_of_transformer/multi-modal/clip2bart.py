

import torch
import torch.nn as nn

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    CLIPVisionModel,
    CLIPImageProcessor,
    GenerationConfig
)

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

'''< 기존구조
graph TD
    A[이미지] --> B[CLIP 인코더]
    B --> C[프로젝션 레이어]
    C --> D[BART 인코더 출력]
    D --> E[Attention Mask 생성]
    E --> F[BART 디코더]
    F --> G[캡션 생성]
'''

''' < 수정방향
graph TD
    A[원본 이미지] --> B[CLIP 인코더]
    B --> C[패치 임베딩 50개]
    C --> D[8개 어텐션 헤드]
    D --> E[헤드별 가중치 분포]
    E --> F[시각화]
'''

'''
graph LR
    A[원본 이미지] --> B[224x224 리사이즈]
    B --> C[패치 분할 32x32]
    C --> D[7x7 어텐션 그리드]
    D --> E[32x32 크로네커 업샘플링]
    E --> F[224x224 어텐션 맵]
'''
    
def plot_attention_map(img, attention_weights, patch_size=32):
    """
    image_path: 원본 이미지 경로
    attention_weights: [num_heads, target_len, source_len]
    patch_size: CLIP의 패치 크기 (ViT-B/32는 32x32)
    """
    # # 원본 이미지 로드
    # img = Image.open(image_path).convert("RGB")
    # img_width, img_height = img.size
    
    # CLIP의 패치 그리드 계산 (예: 224x224 이미지 → 7x7 그리드)
    grid_size = int(np.sqrt(attention_weights.shape[-1] - 1))  # [CLS] 토큰 제외
    
    # 첫 번째 헤드, 첫 번째 쿼리 선택
    attn = attention_weights[0, 0, 1:].cpu().numpy()  # [CLS] 토큰(0번) 제외
    attn = attn.reshape(grid_size, grid_size)
    
    # 어텐션 맵 업샘플링
    attn_resized = np.kron(attn, np.ones((patch_size, patch_size)))
    
    # 시각화
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(attn_resized, cmap='viridis', alpha=0.5)
    plt.imshow(img, alpha=0.5)  # 반투명 오버레이
    plt.title("Attention Map")
    
    plt.show()
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.attention_weights = None  # 어텐션 가중치 저장용

    def forward(self, query, key_value):
        attn_output, attn_weights = self.mha(
            query, key_value, key_value,
            need_weights=True  # 어텐션 가중치 반환 활성화
        )
        self.attention_weights = attn_weights.detach()
        attn_output = self.dropout(attn_output)
        return self.norm(query + attn_output)

class DeepAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 768),
            nn.LayerNorm(768)
        )
        # Cross-Attention
        self.cross_attn = CrossAttentionLayer()

    def forward(self, image_embeds):
        # MLP 프로젝션
        x = self.mlp(image_embeds)  # [B, seq_len, 768]
        
        # 크로스 어텐션 (학습된 쿼리 사용)
        learned_queries = nn.Parameter(torch.randn(1, 32, 768))  # [1, 32, 768]
        queries = learned_queries.expand(x.size(0), -1, -1)
        return self.cross_attn(queries, x)
    
class MultimodalBartModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32",cache_dir='./')
        self.text_decoder = BartForConditionalGeneration.from_pretrained("facebook/bart-base",cache_dir='./')
        self.adapter = DeepAdapter()

        # 가중치 고정
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        vision_outputs = self.vision_encoder(pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state  # [B, 50, 768]
        return self.adapter(image_embeds)

def generate_image_caption(image_path, max_length=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화
    model = MultimodalBartModel().to(device)
    model.eval()
    
    # 프로세서 초기화
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir='./')
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base",cache_dir='./')
    
    # 이미지 처리 < backup
    # image = Image.open(image_path).convert("RGB")
    # inputs = image_processor(images=image, return_tensors="pt").to(device)
    # pixel_values = inputs.pixel_values
        
    
    # 모델별 패치 설정
    patch_config = {
        'clip_vit_b32': {'patch_size': 32, 'img_size': 224},
        'clip_vit_b16': {'patch_size': 16, 'img_size': 224}
    }
    cfg = patch_config['clip_vit_b32']
    
    # 원본 이미지를 모델 입력 크기로 리사이즈
    transform = T.Compose([
        T.Resize(  # 수정 전: T.Resize(cfg['img_size'], ...)
            (cfg['img_size'], cfg['img_size']),  # (height, width) 튜플 필수
            interpolation=InterpolationMode.BILINEAR
        ),
        T.CenterCrop(cfg['img_size']),
        # T.ToTensor(),
        # T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    
    original_img = Image.open(image_path).convert("RGB")
    # 텐서 변환 (채널 x 높이 x 너비)    
    resized_img = transform(original_img)    
    inputs = image_processor(resized_img, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values      
    
    '''
    입력 이미지: 224x224
    패치 크기: 32x32 → 7x7 그리드 (49 패치)
    총 시퀀스 길이: 49 + 1 ([CLS]) = 50
    '''
    # 이미지 임베딩 생성
    with torch.no_grad():
        adapted_embeds = model(pixel_values)        
        attention_mask = torch.ones(
            adapted_embeds.size()[:-1], 
            dtype=torch.long, 
            device=device
        )
        # 어텐션 가중치 추출        
        attn_weights = model.adapter.cross_attn.attention_weights # attn_weights.shape = [batch, num_heads, target_len, source_len]
        # target_len: 학습된 쿼리 개수 (32), source_len: 이미지 패치 수 + 1 ([CLS] 토큰)
    
    # # 첫 번째 배치, 첫 번째 헤드 시각화
    plot_attention_map(resized_img, attn_weights[0].unsqueeze(0)) 
    '''
    #   num_heads=8인 경우 8개의 서로 다른 어텐션 맵 생성
        Head 0: [주로 얼굴 영역 집중]
        Head 1: [카메라 장비 영역 강조]
        Head 2: [배경 건물에 집중]
        ...
        Head 7: [전체적인 구도 파악]
    '''
    # # 모든 헤드 시각화
    # for head_idx in range(attn_weights.size(1)):  # num_heads 차원
    #     plot_attention_map(
    #         resized_img, 
    #         attn_weights[:, head_idx].unsqueeze(1)  # [1, 1, 32, 50]
    #     )
        
    # # 특정 쿼리가 주목하는 패치 찾기 < 쿼리-패치 상관관계 분석
    # query_idx = 5  # 5번 쿼리
    # top_patches = attn_weights[0, 0, query_idx].argsort(descending=True)[:3]
    
    
    # 다중 헤드 어텐션 평균
    # attn_mean = attn_weights.mean(dim=1)  # 헤드 차원 평균
    
    # 동적 오버레이 강도 조절
    # attn_resized = np.clip(attn_resized * 2.5, 0, 1)  # 시각화 강도 조정
    
    '''
    print(tokenizer.special_tokens_map) # {'bos_token': '<s>', 'eos_token': '</s>', ...}    
    # 중간 출력 확인
    print(projected_embeds.shape)  # [1, 50, 768]
    print(attention_mask.shape)    # [1, 50]
    '''
    # 생성 설정
    generation_config = GenerationConfig(
        max_length=max_length,
        # do_sample=True,
        num_beams=5,
        early_stopping=True,
        repetition_penalty=2.0,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        # temperature=0.7,
        top_k=50,
        # top_p=0.95,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # forced_bos_token_id=tokenizer.bos_token_id
    )

    # 캡션 생성
    generated_ids = model.text_decoder.generate(
        inputs_embeds=adapted_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config
    )
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    '''
    성능 향상 기법
        Beam Search: num_beams=4 설정으로 다양한 후보 탐색
        조기 종료: early_stopping=True로 최적 시점에 생성 중단
        특수 토큰 처리: BOS(Begin-of-Sentence), EOS(End-of-Sentence) 토큰 활용
    '''
    

if __name__ == "__main__":
    image_path = "test.jpeg"
    caption = generate_image_caption(image_path)
    print(f"생성된 설명: {caption}")
    
'''

# 학습 루프 예시
optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-4)
for epoch in range(10):
    for images, captions in coco_loader:
        adapted_embeds = model(images)
        outputs = model.text_decoder(inputs_embeds=adapted_embeds, labels=captions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
'''