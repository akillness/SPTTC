import torch
from torch.utils.data import DataLoader
from contrastive_model import ContrastiveModel, contrastive_loss
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPProcessor
import os
from torch.utils.tensorboard import SummaryWriter

# 데이터셋 클래스 예시 (실제 데이터에 맞게 구현 필요)
class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir='./')
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                       std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image).to(torch.float32)
        text = self.texts[idx]
        return image, text

def generate_caption(model, image_path, candidate_texts, transform, device):
    # 이미지 전처리
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # 이미지 임베딩 추출
    with torch.no_grad():
        image_embed = model.image_encoder(image)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    
    # 텍스트 임베딩 일괄 처리
    with torch.no_grad():
        text_embeds = model.text_encoder(candidate_texts)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # 유사도 계산
    similarity = (image_embed @ text_embeds.T).squeeze(0)
    best_idx = similarity.argmax().item()
    return candidate_texts[best_idx]

def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    # 모델 초기화
    model = ContrastiveModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 텐서보드 작성기 초기화
    writer = SummaryWriter()
    os.makedirs("checkpoints", exist_ok=True)
    
    # 데이터셋 예시 (실제 데이터 경로로 변경 필요)
    dataset = ContrastiveDataset(
        image_paths=["sample1.jpg", "sample2.jpg"], 
        texts=["A photo of a cat", "A picture of a dog"]
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 학습 루프
    best_loss = float('inf')
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            
            # 모델 실행
            logits_per_image, logits_per_text = model(images, texts)
            
            # 손실 계산
            loss = contrastive_loss(logits_per_image, logits_per_text)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 텐서보드 로깅
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
            running_loss += loss.item()
            
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 에폭별 평균 손실 기록
        avg_loss = running_loss / len(dataloader)
        writer.add_scalar('Avg Loss/epoch', avg_loss, epoch)
        
        # 모델 저장 (체크포인트)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"checkpoints/model_epoch_{epoch}.pt")
        
        # 베스트 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        
        # 학습 루프 이후에 추가
        if epoch == 9:  # 최종 에폭에서만 예시 실행
            model.eval()
            test_image = "sample1.jpg"  # 실제 이미지 경로로 변경 필요
            candidate_captions = [
                "A photo of a cat",
                "A picture of a dog",
                "A bird sitting on a tree",
                "A car driving on the road"
            ]
            
            generated_caption = generate_caption(
                model=model,
                image_path=test_image,
                candidate_texts=candidate_captions,
                transform=dataset.transform,
                device=device
            )
            print(f"\nTest Image: {test_image}")
            print(f"Generated Caption: {generated_caption}")
    
    writer.close()

# 별도 추론 함수 추가
def infer(model_path, image_path, candidate_texts):
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    # 모델 로드
    model = ContrastiveModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 데이터셋 변형 로드
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                   std=(0.26862954, 0.26130258, 0.27577711))
    ])
    
    # 캡션 생성
    caption = generate_caption(model, image_path, candidate_texts, transform, device)
    print(f"\nInput Image: {image_path}")
    print(f"Predicted Caption: {caption}")

if __name__ == "__main__":
    # 학습 모드
    train()
    
    # 추론 모드 사용 예시 (학습 후 별도 실행 가능)
    # infer(
    #     model_path="checkpoints/best_model.pth",
    #     image_path="test_image.jpg",
    #     candidate_texts=[
    #         "a photo of a cat",
    #         "a picture of a dog",
    #         "a landscape painting",
    #         "a red sports car"
    #     ]
    # ) 