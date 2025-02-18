from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from query.query_pipeline import query_pipeline
from search.search_pipeline import search_pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import VLLM
from langchain.prompts import PromptTemplate
import torch
import re
import dotenv
import yaml
import os
import asyncio
import torch.distributed as dist


def load_model(model_name):
    # 분산 학습 초기화 (예: 4개의 프로세스를 사용)
    dist.init_process_group(backend="nccl", world_size=4, rank=0)
    
    # VLLM 방식으로 모델 로드
    llm = VLLM(model=model_name, trust_remote_code=True, max_new_tokens=512, use_cached_outputs=True)
    return llm # Tokenizer 별도로 로드할 필요 없음.

if __name__ == '__main__':
    # Subquerying & Query Routing 모델: 2.4B EXAONE
    query_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct" 
    query_model = load_model(query_model_name)

    # Summarize & Answer 모델: 7.8B EXAONE
    answer_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    answer_model = load_model(answer_model_name)

    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    dotenv.load_dotenv('~/agent_ai')

    while(True):
        query = input("\n입력 >  ")
        processed_query = query_pipeline(query, query_model)
        
        answer_model.eval()
        pipe = pipeline("text-generation", model=answer_model, max_new_tokens=512)
        
        # LangChain의 LLM으로 Wrapping
        answer_llm = HuggingFacePipeline(pipeline=pipe)
        
        search_result = search_pipeline(processed_query, answer_llm)
        search_result_str = str(search_result)
        search_result_str = re.sub(r"[{}]", "", search_result_str)
        
        answer_prompt = f"""[|system|] 아래 정보를 참고하여 사용자 질문에 답변하세요.
        정보: {search_result_str} [|endofturn|]
        [|user|] 사용자 질문: {query} [|endofturn|]
        [|assistant|]answer"""
        answer_prompt = answer_prompt[:1024]
        
        chat_prompt = PromptTemplate.from_template(answer_prompt)
        print(chat_prompt.input_variables)  # ['query'] 또는 ['text'] 등의 리스트 출력      
        
        # chaining
        chain = chat_prompt | answer_llm | StrOutputParser()
        
        # pipeline 실행
        answer = chain.invoke(query)
        answer = answer.split("[|assistant|]")[1].split("[|endofturn|]")[0].strip()
        
        print("==============Model answer Result (테스트 중, EXAONE 3.5 2.4B)==============")
        print(answer)


    # example_query
    # 이번 주에 부산으로 여행을 가려는데 날씨가 괜찮을까? 또 거기서 가볼만한 곳 추천해줘