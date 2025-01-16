
# << Vision Transformer 기반의 오픈소스 프레임워크 모델
## DETR : https://huggingface.co/facebook/detr-resnet-50
## Deformable DETR < 
## YOLOS (You Only Look One-level Series) <

# ㄴ 위의 두 모델(RT-DETR vs YOLOS) 성능 비교 논문 : https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt

# Load the pre-trained model and processor
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

# Load an image
# url = 'https://example.com/sample-image.jpg'  # Replace with your image URL
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open('./practice/openvino/screenshot.jpg')

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Process the outputs
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
# Draw the results on the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.9:  # Filter out low-confidence detections
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=4)
        draw.text((box[0], box[3]), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="green", font=font)

# Save the image
output_path = 'output_image.jpg'
image.save(output_path)
print(f"Result image saved to {output_path}")

# Optionally, display the image
plt.imshow(image)
plt.axis('off')
plt.show()