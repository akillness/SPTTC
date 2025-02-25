from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from PIL import Image
from io import BytesIO

from huggingface_hub import login

# 생성한 액세스 토큰을 입력합니다.
login(token="")


# 원하는 캐시 디렉토리 경로
custom_cache_dir = './img_caption/'

# 이미지 로드
# image_url = 'https://dfstudio-d420.kxcdn.com/wordpress/wp-content/uploads/2019/06/digital_camera_photo-1080x675.jpg'
image_url = "https://images.unsplash.com/photo-1575936123452-b67c3203c357?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8JTIzaW1hZ2V8ZW58MHx8MHx8fDA%3D"
# image_url = 'https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8dGV4dCUyMG1lc3NhZ2VzfGVufDB8fDB8fHww'
image = Image.open(requests.get(image_url, stream=True).raw)

# 모델 및 프로세서 로드
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base',cache_dir=custom_cache_dir)
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base',cache_dir=custom_cache_dir)

# 입력 데이터 전처리
inputs = processor(image, return_tensors="pt")

# 텍스트 생성
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)