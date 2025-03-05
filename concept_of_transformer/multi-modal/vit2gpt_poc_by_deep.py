import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, GPT2LMHeadModel, GPT2Tokenizer

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
        self.image_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='./')
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='./')
        self.adapter = ImageToTextAdapter()
        
        # 가중치 고정
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.gpt.parameters():
            param.requires_grad = False

def generate_image_caption(image_path, max_length=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalModel().to(device)
    model.eval()
    
    # CLIP 이미지 프로세서 초기화
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir='./')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 이미지 처리 및 임베딩
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        clip_outputs = model.image_encoder(**inputs)
    image_emb = clip_outputs.pooler_output  # CLIP의 pooled 이미지 특징 추출
    adapted_emb = model.adapter(image_emb).unsqueeze(1)

    # GPT 입력 생성
    prompt = "A girl on load with camera"
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        # padding=True,
        # truncation=True,
        return_attention_mask=True
    ).to(device)

    text_emb = model.gpt.transformer.wte(inputs.input_ids)
    
    # 멀티모달 입력 결합
    image_mask = torch.ones(1, 1).to(device)
    attention_mask = torch.cat([image_mask, inputs.attention_mask], dim=1)
    combined_emb = torch.cat([adapted_emb, text_emb], dim=1)

    # 텍스트 생성
    generated_ids = model.gpt.generate(
        inputs_embeds=combined_emb,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=1,
        temperature=0.1,
        early_stopping=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    image_path = "test.jpeg"
    caption = generate_image_caption(image_path)
    print(f"생성된 설명: {caption}")