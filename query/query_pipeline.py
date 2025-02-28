import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from query.query_with_gemma2 import get_sub_queries
from query.query_routing_rule import rule_based_routing
from query.parallel import prompt_routing
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.llms import VLLM
from langchain_huggingface import HuggingFacePipeline
from my_utils import timeit

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
        vllm_kwargs={"max_model_len": 10000}
        )
    return llm

@timeit
def query_pipeline(query: str, llm) -> tuple[list, list[dict]]:
    """
    서브쿼리 분해와 쿼리 라우팅을 수행하는 파이프라인
    args:
        query: str, 입력 쿼리
        llm: LangChain.VLLM, 사용할 llm
    return:
        subqueries: list, 분해된 서브쿼리
        final_processed_query: list[Dict], 라우팅 최종 처리된 쿼리
    """
    # 서브쿼리 분해
    subqueries = get_sub_queries(query, llm)
    print("fetched sub queries")
    print(subqueries)   # for debugging

    # 쿼리 라우팅
    # rule-based routing
    processed_query = []
    to_llm_subqueries = []
    for subquery in subqueries:
        routing = rule_based_routing(subquery)
        if routing == 'llm':
            to_llm_subqueries.append(subquery)
        else:   # 'web', 'none'
            processed_query.append({'subquery': subquery, 'routing': routing})

    # llm-based routing
    result = prompt_routing(to_llm_subqueries, llm)
    llm_processed_query = []
    for res in result:
        llm_processed_query.append({'subquery': res['subquery'], 'routing': res['routing']})
    final_processed_query = llm_processed_query + processed_query
    print(final_processed_query)

    return subqueries, final_processed_query

    

if __name__ == '__main__':
    # 인자 설정

    load_func = load_vllm #if args.vllm == 'true' else load_model

    # MODEL_NAME = "recoilme/recoilme-gemma-2-9B-v0.4"
    # MODEL_NAME = "beomi/gemma-ko-7b"
    # MODEL_NAME = "google/gemma-2-2b-it"
    MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    llm = load_func(MODEL_NAME)

    query = input("입력 >  ")
    query_pipeline(query, MODEL_NAME, llm)