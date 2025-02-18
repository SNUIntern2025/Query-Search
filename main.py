from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from query.query_pipeline import query_pipeline
from search.search_pipeline import search_pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
from langchain_community.llms import VLLM
from final_output import final_output
import argparse
from datetime import datetime
import os
from dotenv import load_dotenv
from huggingface_hub import login

#연구실 모델 사용을 위한 토큰 설정
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


def load_model(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True, trust_remote_code=True)
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    # LangChain의 LLM으로 Wrapping
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def load_vllm_1(MODEL_NAME):
    # (옵션1) vLLM을 백엔드 서버로 띄우고 Langchain이 그 API를 호출하는 방식
    from langchain_community.llms import VLLMOpenAI
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",  # 로컬 vLLM 서버 주소
        api_key="token-snuintern2025"
    )

    llm = VLLMOpenAI(
        openai_api_key= "token-snuintern2025",
        openai_api_base="http://localhost:8000/v1", # 로컬 서버에 띄워서 백엔드에서 실행하기
        model_name=MODEL_NAME,
        max_tokens = 1024, # 모델마다 달라질 수 있음
        temperature = 0.7,
        streaming = True,
        top_p=0.85
    )

    return llm

def load_vllm_2(MODEL_NAME):
    #(옵션2) 로컬에서 모델을 사용하는 방법
    llm = VLLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        top_k=3,
        top_p=0.85,
        temperature=0.2,
        do_sample=True,
        repitition_penalty=1.2,
        tensor_parallel_size=4,
        vllm_kwargs={"max_model_len": 5000}
        )

    return llm


if __name__ == '__main__':
    # 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm', type=str, default='true', help='Using vLLM or not')
    args = parser.parse_args()
    #load_vllm_1 또는 load_vllm_2 선택하여 코드 수정 필요!
    load_func = load_vllm_1 if args.vllm == 'true' else load_model

    #사용 모델
    MODEL_NAME = "snunlp/bigdata_gemma2_9b_dora"

    llm = load_func(MODEL_NAME)

    while(True):
        query = input("\n입력 >  ")
        try:
            start = datetime.now()
            _, processed_query = query_pipeline(query, MODEL_NAME, llm, args.vllm)
            search_result = search_pipeline(processed_query, llm, args.vllm)

            print("\n\n==============Model answer==============\n")
            answer = final_output(query, search_result, llm)
            print(answer)
            end = datetime.now()
            print("총 소요 시간: ", end-start)
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}. try again.")


    # example_query
    # 이번 주 토요일에 부산으로 여행을 가려는데 날씨가 괜찮을까? 또 거기서 가볼만한 곳 추천해줘