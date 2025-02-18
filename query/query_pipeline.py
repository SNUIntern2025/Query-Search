import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from query.query_with_gemma2 import get_sub_queries
from query.query_routing_rule import rule_based_routing
from query.parallel import prompt_routing
from my_utils import timeit


@timeit
def query_pipeline(query, model_name, llm, is_vllm):
    """
    query와 관련된 전반적인 파이프라인을 담당하는 함수.
    args:
        query: 사용자 입력 쿼리
        model_name: 모델 이름
        llm: 사용할 LLM
        is_vllm: vLLM 사용 여부
        
    return: 
        subqueries: 서브쿼리 리스트
        final_processed_query: 최종 처리된 쿼리 리스트
    """
    # 서브쿼리 분해
    subqueries = get_sub_queries(query, llm, model_name)

    # 쿼리 라우팅
    # 1. rule-based routing
    processed_query = []
    to_llm_subqueries = []
    for subquery in subqueries:
        routing = rule_based_routing(subquery)
        if routing == 'llm':
            to_llm_subqueries.append(subquery)
        else:   # 'web', 'none'
            processed_query.append({'subquery': subquery, 'routing': routing})

    # 2. llm-based routing
    result = prompt_routing(to_llm_subqueries, llm, model_name, is_vllm)
    llm_processed_query = []
    for res in result:
        llm_processed_query.append({'subquery': res['subquery'], 'routing': res['routing']})
    final_processed_query = llm_processed_query + processed_query

    return subqueries, final_processed_query

    
# ================================= Test =================================
# if __name__ == '__main__':
#     # 인자 설정
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--vllm', type=str, default='true', help='Using vLLM or not')
#     args = parser.parse_args()

#     load_func = load_vllm if args.vllm == 'true' else load_model

#     # MODEL_NAME = "recoilme/recoilme-gemma-2-9B-v0.4"
#     # MODEL_NAME = "beomi/gemma-ko-7b"
#     # MODEL_NAME = "google/gemma-2-2b-it"
#     MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

#     llm = load_func(MODEL_NAME)

#     query = input("입력 >  ")
#     query_pipeline(query, MODEL_NAME, llm, args.vllm)