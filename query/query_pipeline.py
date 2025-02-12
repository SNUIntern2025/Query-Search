from query.query_with_gemma2 import get_sub_queries
from query.query_routing_rule import rule_based_routing
from query.parallel import prompt_routing
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True)
    return tokenizer, model


def query_pipeline(query, tokenizer, model):
    # query = input("입력 >  ")

    # 서브쿼리 분해
    subqueries = get_sub_queries(query, tokenizer, model)
    print("\n\n==============Sub Querying Result (Gemma2-9B)==============\n")
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
        result = prompt_routing(to_llm_subqueries)
        llm_processed_query = []
        for res in result:
            llm_processed_query.append({'subquery': res['subquery'], 'routing': res['routing']})
        final_processed_query = llm_processed_query + processed_query
    else:
        final_processed_query = processed_query

    print("\n\n==============Query routing Result (rule-based + gpt-4o-mini) ==============\n")
    print(final_processed_query)

    return final_processed_query


if __name__ == '__main__':
    MODEL_NAME = "recoilme/recoilme-gemma-2-9B-v0.4"
    tokenizer, model = load_model(MODEL_NAME)

    query = input("입력 >  ")
    query_pipeline(query, tokenizer, model)