import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import asyncio
from my_utils import timeit
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer
from query.query_with_gemma2 import special_tokens
import re

async def summarize(docs, llm, is_vllm, max_tokens=1500, max_concurrent_tasks=8, model_name="snunlp/bigdata_gemma2_9b_dora"):
        
    """HuggingFace LLM으로 비동기적 요약을 실행"""
    if not docs:
        return {}
    summary_length = max_tokens / len(docs)

    system_prompt = f"""다음 텍스트를 읽고, {max(int(summary_length/40), 5)}문장 이내로 요약해주세요.
        텍스트에 언급된 중요한 고유 명사는 포함되어야 합니다.
        외부 정보를 포함하지 않고 제공된 텍스트에만 전적으로 의존하세요."""

    template = f"""{special_tokens[model_name]["system_start"]} {system_prompt}
    {special_tokens[model_name]["end_token"]}

    {special_tokens[model_name]["user_start"]} {{text}} {special_tokens[model_name]["end_token"]}

    {special_tokens[model_name]["assistant_start"]} """

    prompt_template = PromptTemplate.from_template(template)

    # 요약 체인 로딩
    chain = (
    prompt_template
    | llm
    | StrOutputParser()
    )

    # 비동기 작업의 수를 제한하는 세마포어를 설정하여 동시에 실행되는 작업의 수를 max_concurrent_tasks로 제한합니다.
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    # 각 문서를 받아온 후 요약 작업을 수행하는 비동기 함수입니다
    # vllm과의 충돌 문제 때문에, 우선 ainvoke 제외하여 실행하였습니다.
    async def summarize_task(doc):
        async with semaphore:
            result = await chain.ainvoke({"text": doc})
            result = result.split(special_tokens[model_name]["assistant_start"])[-1].strip()
            return result

            
    # doc의 길이에 따라 문서를 처리합니다
    split_docs = []
    contexts = []
    for doc in docs:
        shortened_doc = doc[:1500]
        if len(shortened_doc) < 1500/len(docs):
            print("Summarizer: Doc added directly to context, we have saved some GPU")
            contexts.append(shortened_doc) # 요약문의 예상 길이보다 doc이 짧은 경우, 요약 없이 final_output으로 보냅니다.
        else:
            print("Summarizer: Doc added to summarizer")
            split_docs.append(shortened_doc) # doc의 첫 n개 토큰을 요약할 문서 리스트에 넣습니다
    
    if is_vllm == 'false':  # vllm이 아닐 경우, 비동기적으로 요약 작업을 실행합니다
        summaries = await asyncio.gather(*[summarize_task(doc) for doc in split_docs])
    else:   # vllm일 경우 비동기를 제거하고 batch 단위로 함수를 실행합니다.
        summaries = chain.batch([{"text": doc} for doc in split_docs])
        summaries = [re.sub(r'\n', ' ', summary) for summary in summaries]

    return contexts + summaries  # 요약된 결과를 리스트 형태로 반환합니다.
