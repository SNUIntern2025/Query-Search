from query.exaone_pipe import get_sub_queries
from query.query_routing_rule import rule_based_routing
from query.parallel import prompt_routing
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



def query_pipeline(query, model):
    # query = input("입력 >  ")

    # 서브쿼리 분해
    subqueries = get_sub_queries(query, model)
    print("\n\n==============Sub Querying Result (Exaone-3.5-2.4B)==============\n")
    print(subqueries)


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
    if len(to_llm_subqueries) > 0:
        result = prompt_routing(to_llm_subqueries, model) # parallel.py
        llm_processed_query = []
        for res in result:
            llm_processed_query.append({'subquery': res['subquery'], 'routing': res['routing']})
        final_processed_query = llm_processed_query + processed_query
    else:
        final_processed_query = processed_query

    print("\n\n==============Query routing Result (rule-based + gemma-ko-7b) ==============\n")
    print(final_processed_query)

    return final_processed_query