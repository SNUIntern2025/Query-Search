from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from query.query_pipeline import query_pipeline
from search.search_pipeline import search_pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
from langchain_community.llms import VLLM
from final_output import final_output
import argparse
from datetime import datetime


def load_model(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True, trust_remote_code=True)
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    # LangChain의 LLM으로 Wrapping
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def load_vllm(MODEL_NAME):
    llm = VLLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_new_tokens=512,
        top_k=3,
        top_p=0.85,
        temperature=0.2,
        do_sample=True,
        repitition_penalty=1.2,
        )
    return llm


if __name__ == '__main__':
    # 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm', type=str, default='true', help='Using vLLM or not')
    args = parser.parse_args()
    load_func = load_vllm if args.vllm == 'true' else load_model

    # 모델 및 환경 세팅
    # MODEL_NAME = "recoilme/recoilme-gemma-2-9B-v0.4"
    # MODEL_NAME = "beomi/gemma-ko-7b"
    # MODEL_NAME = "google/gemma-2-2b-it"
    MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    llm = load_func(MODEL_NAME)

    while(True):
        query = input("\n입력 >  ")
        try:
            start = datetime.now()
            processed_query = query_pipeline(query, MODEL_NAME, llm, args.vllm)
            search_result = search_pipeline(processed_query, llm, args.vllm)

            print("\n\n==============Model answer==============\n")
            answer = final_output(query, search_result, llm)
            print(answer)
            end = datetime.now()
            print("총 소요 시간: ", end-start)
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}. try again.")


    # example_query
    # 이번 주에 부산으로 여행을 가려는데 날씨가 괜찮을까? 또 거기서 가볼만한 곳 추천해줘