import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from query.query_with_gemma2 import get_sub_queries
from query.query_routing_rule import rule_based_routing
from query.parallel import prompt_routing
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.llms import VLLM
from langchain_huggingface import HuggingFacePipeline
import argparse
from my_utils import timeit

# for not using vllm
def load_model(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # for flash attention 2
    # model = AutoModelForCausalLM.from_pretrained(
    # MODEL_NAME, 
    # torch_dtype=torch.float16,
    # device_map="auto",
    # attn_implementation="flash_attention_2",
    # trust_remote_code=True)

    # for vanilla
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
        max_new_tokens=128,
        top_k=3,
        top_p=0.85,
        temperature=0.2,
        do_sample=True,
        repitition_penalty=1.2,
        )
    return llm


@timeit
def query_pipeline(query, model_name, llm, is_vllm):
    # 서브쿼리 분해
    print("==============Sub Querying Result==============")
    subqueries = get_sub_queries(query, llm, model_name)
    print(subqueries)


    # 쿼리 라우팅
    # rule-based routing
    print("==============Query routing Result==============")
    processed_query = []
    to_llm_subqueries = []
    for subquery in subqueries:
        routing = rule_based_routing(subquery)
        if routing == 'llm':
            to_llm_subqueries.append(subquery)
        else:   # 'web', 'none'
            processed_query.append({'subquery': subquery, 'routing': routing})

    # llm-based routing
    result = prompt_routing(to_llm_subqueries, llm, model_name, is_vllm)
    llm_processed_query = []
    for res in result:
        llm_processed_query.append({'subquery': res['subquery'], 'routing': res['routing']})
    final_processed_query = llm_processed_query + processed_query
    
    print(final_processed_query)

    return final_processed_query

    

if __name__ == '__main__':
    # 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm', type=str, default='true', help='Using vLLM or not')
    args = parser.parse_args()

    load_func = load_vllm if args.vllm == 'true' else load_model

    # MODEL_NAME = "recoilme/recoilme-gemma-2-9B-v0.4"
    # MODEL_NAME = "beomi/gemma-ko-7b"
    # MODEL_NAME = "google/gemma-2-2b-it"
    MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    llm = load_func(MODEL_NAME)

    query = input("입력 >  ")
    query_pipeline(query, MODEL_NAME, llm, args.vllm)