import torch
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from query.subquerying_prompt import SYSTEM_GEMMA, SYSTEM_EXAONE
from query.few_shot import examples_final
import json
from datetime import datetime
from my_utils import timeit


special_tokens = {  # 채은님 코드 빌려오기
    "snunlp/bigdata_gemma2_9b_dora": { # snunlp 모델로 변경했음.
        "system_start": "<start_of_turn>system",
        "user_start": "<start_of_turn>user",
        "assistant_start": "<start_of_turn>model",
        "examples_start": "<start_of_turn>example",
        "end_token": "<end_of_turn>"
    },
    "snunlp/bigdata_exaone3_7.8b_fft": { # snunlp 모델로 변경했음.
        "system_start": "[|system|]",
        "user_start": "[|user|]",
        "assistant_start": "[|assistant|]",
        "examples_start": "[|example|]",
        "end_token": "[|endofturn|]"
    },
    "recoilme/recoilme-gemma-2-9B-v0.4": { # 변경 전 gemma 9B 모델
        "system_start": "<start_of_turn>system",
        "user_start": "<start_of_turn>user",
        "assistant_start": "<start_of_turn>model",
        "examples_start": "<start_of_turn>example",
        "end_token": "<end_of_turn>"
    },
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct": { # 변경 전 EXAONE 2.4B 모델
        "system_start": "[|system|]",
        "user_start": "[|user|]",
        "assistant_start": "[|assistant|]",
        "examples_start": "[|example|]",
        "end_token": "[|endofturn|]"
    },
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": { # 변경 전 EXAONE 7.8B 모델
        "system_start": "[|system|]",
        "user_start": "[|user|]",
        "assistant_start": "[|assistant|]",
        "examples_start": "[|example|]",
        "end_token": "[|endofturn|]"
    }
}

def load_prompt(system_prompt :str, model_name: str, fewshot_ex=None) -> PromptTemplate:
    '''
    프롬프트를 받아, LangChain의 프롬프트 클래스를 생성하여 반환하는 함수
    args:
        system_prompt: 시스템 프롬프트
        fewshot_ex: list, few-shot 학습 데이터
    return:
        chat_prompt: PromptTemplate
    '''

    today = datetime.today().strftime("%Y년 %m월 %d일")
    today_prompt = f"오늘은 {today}입니다. 올해는 {datetime.today().year}년입니다. 이번 달은 {datetime.today().month}월입니다."

    assistent_template = f"""다음 질문을 여러 개의 하위 질문으로 나누어주세요.
    {special_tokens[model_name]["user_start"]} {{query}} {special_tokens[model_name]["end_token"]}
    {special_tokens[model_name]["assistant_start"]}
    """
    
    if fewshot_ex is not None:
        few_shot_prompt = f"""{special_tokens[model_name]["examples_start"]} 아래는 몇 가지 예시입니다. 아래 예시를 참고해서 답변을 작성하세요.""" \
        + '\n'.join([f"예시 입력: {ex['input']}\n예시 출력: {ex['output']}\n" for ex in fewshot_ex]) + special_tokens[model_name]["end_token"]
        chat_prompt = PromptTemplate.from_template(today_prompt + system_prompt + few_shot_prompt + '\n\n' + assistent_template)

    else:
        chat_prompt = PromptTemplate.from_template(today_prompt + system_prompt + '\n\n' + assistent_template)

    return chat_prompt


@timeit
def get_sub_queries(query: str, llm) -> list[str]:
    '''
    사용자 입력을 받아, 하위 쿼리로 나누어 반환하는 함수
    args:
        query: 사용자 입력
    return:
        sub_queries: 하위 쿼리
    '''

    model_name = llm.model
    # 프롬프트 설정
    if model_name == "recoilme/recoilme-gemma-2-9B-v0.4":
        chat_prompt = load_prompt(SYSTEM_GEMMA, model_name, examples_final)
    else:
        chat_prompt = load_prompt(SYSTEM_EXAONE, model_name, examples_final)

    print("prompt fetched")
    # chaining
    chain = chat_prompt | llm | StrOutputParser()
    print("chained")

    # pipeline 실행
    sub_queries = chain.invoke({"query": query})
    sub_queries = sub_queries.split(special_tokens[model_name]["assistant_start"])[-1].strip()

    # json으로 변환
    sub_queries = json.loads(sub_queries)

    return sub_queries['response']