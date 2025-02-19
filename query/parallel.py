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

def process_single_query(query: str, chain, model_name: str) -> Dict:
    '''
    멀티쓰레딩 작업을 batch 작업으로 변경. 따라서 이 함수는 현재 사용하지 않습니다.
    단일 쿼리를 처리하는 함수
    query: str, 처리할 단일 쿼리
    chain: LangChain chain, 사용할 체인
    '''
    try:
        result = chain.invoke({"current_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"user_input":query})
        result = re.sub(r'```json', '', result)
        result = result.split(special_tokens[model_name]["assistant_start"])[-1].strip()
        print(result)
        return result
    except Exception as e:
        print(f"Error processing query '{query}': {str(e)}")
        return f"""{{
            "subquery": query,
            "routing": "web",
            "reasoning": "Error processing: {str(e)}"
        }}"""

# 라우팅 결과를 후처리하는 함수
def post_process_result(model_name, text):
    text = re.sub(r'(```|```json|json\n)', '', text)
    text = text.split(special_tokens[model_name]["assistant_start"])[-1].strip()
    json_result = json.loads(text)
    return json_result

@timeit
def prompt_routing(subqueries: List[str], llm, is_vllm):
    '''
    subquery를 받아, LLM prompting을 통해 routing을 병렬로 실행하는 함수
    subqueries: str[], 사용자 입력을 subquery로 분해한 list
    '''
    model_name = llm.model

    if 'gemma' in model_name:
        # parallel_index 제외한 partial 함수
        chat_prompt = load_prompt(PARALLEL_GEMMA, model_name)
    elif 'EXAONE' in model_name or 'exaone' in model_name:
        chat_prompt = load_prompt_exaone(PARALLEL_EXAONE, model_name, examples_routing)

    chain = (chat_prompt 
            | llm 
            | StrOutputParser())
    
    result = chain.batch([{"current_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user_input": query} for query in subqueries])
    all_responses = list(map(partial(post_process_result, model_name), result))
    print(all_responses)
    
    return all_responses