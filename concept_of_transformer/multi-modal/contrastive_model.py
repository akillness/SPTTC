import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", cache_dir='./')
        
    def forward(self, x):
        outputs = self.clip(x)
        return outputs.image_embeds

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", cache_dir='./')
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir='./')
        
    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.clip(**inputs.to(self.clip.device))
        return outputs.text_embeds

class ContrastiveModel(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1.0 / temperature))
        # MPS 디바이스 호환성을 위한 수정
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        self.to(self.device)
        
    def forward(self, images, texts):
        images = images.to(self.device)
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(texts)
        
        # 정규화
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # 유사도 계산
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

def contrastive_loss(logits_per_image, logits_per_text):
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
    return (loss_i + loss_t) / 2 