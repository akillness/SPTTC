import torch
from transformers import pipeline

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")


## Sementic 에 대한 감성 분석
# classifier = pipeline("sentiment-analysis")

# results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
# for result in results:
#     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


## mask 단어에 대한 컨텍스트 생성
# unmasker = pipeline("fill-mask")
# results = unmasker("This <mask> will teach you all about models",top_k =4)

# for result in results:
#     print(result['sequence'])

ner = pipeline("ner", grouped_entities=True)
print(ner("I'm Jangyoung and work at NCSOFT in Pangyo"))

