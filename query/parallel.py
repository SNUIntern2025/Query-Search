from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from query.routing_prompts import *
from query.few_shot import examples_routing
import json
from datetime import datetime
from typing import List, Dict
from query.query_with_gemma2 import special_tokens
from my_utils import timeit
import re
from functools import partial

# from queries import SAMPLE_QUERIES

def load_prompt(system_prompt: str, model_name) -> PromptTemplate:
    '''
    프롬프트를 받아, LangChain의 PromptTemplate 클래스를 생성하여 반환하는 함수
    system_prompt: str, 시스템 프롬프트
    '''
    template = f"""{special_tokens[model_name]["system_start"]} {system_prompt}
    현재 시각: {{current_time}} {special_tokens[model_name]["end_token"]}

    {special_tokens[model_name]["user_start"]} {{user_input}} {special_tokens[model_name]["end_token"]}

    {special_tokens[model_name]["assistant_start"]} """
    return PromptTemplate.from_template(template)

# 엑사원이 프롬프트의 말을 듣지 않아 부득이 프롬프트를 교체합니다... 열심히 짜주셨는데 죄송해요 지안님 ㅠㅠ
# 그래도 지안님 프롬프트 참고하여 열심히 수정하였습니다!!
def load_prompt_exaone(system_prompt :str, model_name: str, fewshot_ex=None) -> PromptTemplate:
    '''
    프롬프트를 받아, LangChain의 프롬프트 클래스를 생성하여 반환하는 함수
    args:
        system_prompt: 시스템 프롬프트
        fewshot_ex: list, few-shot 학습 데이터
    return:
        chat_prompt: PromptTemplate
    '''
    assistent_template = f"""{special_tokens[model_name]["user_start"]} {{user_input}} {special_tokens[model_name]["end_token"]}
    {special_tokens[model_name]["assistant_start"]}
    """
    
    if fewshot_ex is not None:
        few_shot_prompt = f"""{special_tokens[model_name]["examples_start"]} 아래는 몇 가지 예시입니다.""" \
        + '\n'.join([f"예시 입력: {ex['input']}\n예시 출력: {ex['output']}\n" for ex in fewshot_ex]) + special_tokens[model_name]["end_token"]
        chat_prompt = PromptTemplate.from_template(system_prompt + few_shot_prompt + '\n\n' + assistent_template)

    else:
        chat_prompt = PromptTemplate.from_template(system_prompt + '\n\n' + assistent_template)

    return chat_prompt

# 라우팅 결과를 후처리하는 함수
def post_process_result(subquery, text: str) -> Dict:
    if re.search(r'"routing"\s*:\s*"web"', text):
        return {"subquery": subquery, "routing": "web", "reasoning": "web routing"}
    elif re.search(r'"routing"\s*:\s*"none"', text):
        return {"subquery": subquery, "routing": "none", "reasoning": "none routing"}
    else:
        return {"subquery": subquery, "routing": "web", "reasoning": "error"}

def prompt_routing(subqueries: List[str], llm, is_vllm):
    '''
    subquery를 받아, LLM prompting을 통해 routing을 병렬로 실행하는 함수
    subqueries: str[], 사용자 입력을 subquery로 분해한 list
    '''
    if hasattr(llm, "model_name"):
        model_name = llm.model_name
    else:
        model_name = llm.model

    if 'EXAONE' in model_name or 'exaone' in model_name:    # 엑사원
        chat_prompt = load_prompt_exaone(PARALLEL_EXAONE, model_name, examples_routing)
    else:   # 나머지 모델
        chat_prompt = load_prompt(PARALLEL_GEMMA, model_name)

    chain = (chat_prompt 
            | llm 
            | StrOutputParser())
    
    result = chain.batch([{"current_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user_input": query} for query in subqueries])
    all_responses = [post_process_result(subquery, res) for subquery, res in zip (subqueries, result)]
    
    return all_responses
