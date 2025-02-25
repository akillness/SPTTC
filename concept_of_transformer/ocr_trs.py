
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# from huggingface_hub import login

# # 생성한 액세스 토큰을 입력합니다.
# login(token="")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

image = 'https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8dGV4dCUyMG1lc3NhZ2VzfGVufDB8fDB8fHww'
inputs = processor(image, return_tensors="pt").to(device)

generate_ids = model.generate(
    **inputs,
    do_sample=False,
    tokenizer=processor.tokenizer,
    stop_strings="<|im_end|>",
    max_new_tokens=4096,
)

find_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(find_text)
