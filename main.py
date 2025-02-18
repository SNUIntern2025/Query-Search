from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from query.query_pipeline import query_pipeline
from search.search_pipeline import search_pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
from langchain_community.llms import VLLM
from langchain_community.llms import VLLMOpenAI # OpenAI-Compatible Completion (서버에 올릴 때 필요)
from final_output import final_output
import argparse
from datetime import datetime
# from MyVLLM import MyVLLM
import os
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

client = OpenAI(
        base_url="http://localhost:8000/v1",  # 로컬 vLLM 서버 주소
        api_key="token-snuintern2025"
    )
 
def load_model(MODEL_NAME): 
    """vLLM 없이 기본으로 모델 로드하는 함수"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True, trust_remote_code=True)
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    # LangChain의 LLM으로 Wrapping
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def load_vllm(MODEL_NAME):
    # 인자 수정하려면 아래 참조
    # /home/hyeznee/.conda/envs/가상환경명/lib/python3.11(파이썬버전명)/site-packages/langchain_community/llms/vllm.py
    """vLLM을 이용하여 모델을 로드하는 함수""" # 로컬에서 vLLM 사용, 혜진님 코드 참조하는 것이 나음.
    # llm = VLLM(
    #     model=MODEL_NAME,
    #     trust_remote_code=True,
    #     max_new_tokens=512,
    #     top_k=3,
    #     top_p=0.85,
    #     temperature=0.2,
    #     do_sample=True,
    #     repitition_penalty=1.2,
    #     gpu_memory_utilization=0.9, # GPU에 할당할 메모리
    #     )
    
    llm = VLLMOpenAI(
        openai_api_key= "token-snuintern2025",
        openai_api_base="http://localhost:8000/v1", # 로컬 서버에 띄워서 백엔드에서 실행하기
        model_name=MODEL_NAME,
        max_tokens = 4096, # 모델마다 달라질 수 있음
        temperature = 0.7,
        streaming = True,
        top_p=0.85
    )
    
    return llm


if __name__ == '__main__':
    
    # 명령행 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm', type=str, default='true', help='Using vLLM or not')
    args = parser.parse_args()
    load_func = load_vllm if args.vllm == 'true' else load_model # vLLM 사용 여부에 따라 모델 로드 함수 변경

    # 모델 및 환경 세팅
    # MODEL_NAME = "recoilme/recoilme-gemma-2-9B-v0.4"
    # MODEL_NAME = "beomi/gemma-ko-7b"
    # MODEL_NAME = "google/gemma-2-2b-it"
    # MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    # MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    
    # 사용해 볼 연구실 모델
    MODEL_NAME = "snunlp/bigdata_gemma2_9b_dora" 
    # MODEL_NAME = "snunlp/bigdata_exaone3_7.8b_fft"
    
    llm = load_func(MODEL_NAME)

while(True):
    query = input("\n입력 >  ")
    # try:
    start = datetime.now()
    _, processed_query = query_pipeline(query, MODEL_NAME, llm, args.vllm)
    
    search_result = search_pipeline(processed_query, llm, args.vllm)

    print("\n\n==============Model answer==============\n")
    answer = final_output(query, search_result, llm)
    print(answer)
    end = datetime.now()
    print("총 소요 시간: ", end - start)
            
    # except Exception as e:
    #     print(f"Error processing query '{query}': {str(e)}. try again.")


    # example_query
    # 이번 주 토요일에 부산으로 여행을 가려는데 날씨가 괜찮을까? 또 거기서 가볼만한 곳 추천해줘