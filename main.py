from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from query.query_pipeline import query_pipeline
from search.search_pipeline import search_pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import torch
import re
import dotenv
import openai
import yaml
import os


def load_model(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True)
    return tokenizer, model


if __name__ == '__main__':
    # 모델 및 환경 세팅
    MODEL_NAME = "recoilme/recoilme-gemma-2-9B-v0.4"
    tokenizer, model = load_model(MODEL_NAME)
    # openai key 받아오기
    # 모델 교체되면 삭제할 것!!!
    with open('../jo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    dotenv.load_dotenv('~/agent_ai')
    # OpenAI API 키 설정
    openai.api_key = config.get('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = openai.api_key

    while(True):
        query = input("\n입력 >  ")
        processed_query = query_pipeline(query, tokenizer, model)
        search_result = search_pipeline(processed_query)

        # 임시
        print("\n========================================")
        print("여기부터는 pipeline 결합을 위해 임시로 구현해놓은 코드입니다.")
        print("아직 최적화가 덜 되었으며, 실행 시간이 느릴 수 있습니다.")
        print("========================================\n")
        # TODO: Langchain으로 바꿔서 연결하기, 프롬프트 구상
        # dictionary to str
        search_result_str = str(search_result)
        search_result_str = re.sub(r"[{}]", "", search_result_str)
        prompt = f"""<|im_start|>system 아래 정보들을 요약하고 참고하여 사용자 질문에 답변하세요.
        정보: {search_result_str} <|im_end|>
        <|im_start|>user 사용자 질문: {query} <|im_end|>
        <|im_start|>answer """
        # prompt = prompt[:2000]  # 일단 max length 이슈로...
        chat_prompt = PromptTemplate.from_template(prompt)

        # huggingface gemma2
        # model.eval()
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        # # LangChain의 LLM으로 Wrapping
        # llm = HuggingFacePipeline(pipeline=pipe)

        # gpt-4o-mini
        llm = ChatOpenAI(model='gpt-4o-mini')

        # chaining
        chain = chat_prompt | llm | StrOutputParser()

        # pipeline 실행
        answer = chain.invoke({"query": query})
        # answer = answer.split("<|im_start|>answer")[1].split("<|im_end|>")[0].strip()

        print("==============Model answer Result (임시, gpt-4o-mini)==============")
        print(answer)


    # example_query
    # 이번 주에 부산으로 여행을 가려는데 날씨가 괜찮을까? 또 거기서 가볼만한 곳 추천해줘