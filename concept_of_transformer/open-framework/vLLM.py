from langchain_community.llms import VLLM  # ✅ 수정된 임포트
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# vLLM 초기화
llm = VLLM(
    model="meta-llama/Llama-3-8B-Instruct",
    trust_remote_code=True,
    max_new_tokens=128
)

# 체인 실행
prompt = PromptTemplate(
    input_variables=["input"],
    template="[INST] 다음을 반말로 변환: {input} [/INST]"
)
# 3. 체인 생성 및 실행
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.invoke({"input": "오늘 날씨가 매우 좋습니다."})["text"])
# 출력: "실례합니다. 이 작업을 수행하는 방법을 알려주시겠습니까?"