import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from transformers import AutoTokenizer

import asyncio
import torch.distributed as dist
import atexit
from langchain.text_splitter import RecursiveCharacterTextSplitter # 추가
from my_utils import timeit

def truncate(doc, count=1000, model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"):
    '''
    doc의 첫 count개의 token만 잘라내는 함수입니다. (02-14 추가)
    Tokenizer encoding은 ms단위로 소요되기 때문에, 일단 사용했습니다.
    doc[str]: 토큰을 셀 문서
    count[int]: 
    model_name[str]: 사용할 tokenizer의 모델 이름. 기본값: 엑사원
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(doc, add_special_tokens=True)[:count]
    return tokenizer.decode(tokens)


# 문서 분할기 (추가한 부분)
"""
def split_docs_by_token_count(docs, max_tokens):
    '''LLM의 최대 토큰 수에 맞춰 문서 분할
    02-14(지안): 토큰 수를 제한하는 로직을 구현할 경우 더 이상 필요하지 않은 함수가 될 것 같습니다.
    '''
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=50)
    split_docs = []
    for doc in docs:
        split_docs.extend(splitter.split_text(doc))
    return split_docs
"""

async def summarize(docs, llm, is_vllm, max_tokens=1000, max_concurrent_tasks=5):
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        
    """HuggingFace LLM으로 비동기적 요약을 실행"""
    if not docs:
        return {}

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""다음 텍스트를 읽고, 주요 아이디어와 핵심 내용을 파악하여, 간결하면서도 명확한 요약문을 작성해 주세요.

        요약문은 다음 항목들을 포함해야 합니다:
        1. 문서의 주요 주제 및 목적
        2. 핵심 내용과 중요한 세부사항
        3. 결론 및 전달하고자 하는 메시지

        외부 정보를 포함하지 않고 제공된 텍스트에만 전적으로 의존하세요.
        
        {text}"""
    )

    # 요약 체인 로딩
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False)

    # 비동기 작업의 수를 제한하는 세마포어를 설정하여 동시에 실행되는 작업의 수를 max_concurrent_tasks로 제한합니다.
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # 문서 분할
    # split_docs = split_docs_by_token_count(docs, max_tokens) # 02-14 (지안): 토큰 수를 제한하는 로직을 구현할 경우, 불필요해집니다.
    
    # 각 문서를 받아온 후 요약 작업을 수행하는 비동기 함수입니다
    # vllm과의 충돌 문제 때문에, 우선 ainvoke 제외하여 실행하였습니다.
    async def summarize_task(doc):
        async with semaphore:
            if is_vllm == 'false':
                return await chain.ainvoke({"input_documents": [Document(page_content=doc)]})
            else:   # 비동기 제거
                return chain.invoke({"input_documents": [Document(page_content=doc)]})
            
    # doc의 길이에 따라 문서를 처리합니다
    split_docs = []
    contexts = []
    for doc in docs:
        shortened_doc = truncate(doc, 500)
        if len(shortened_doc) < 5000:
            print("Summarizer: Doc added directly to context, we have saved some GPU")
            invoke_format = {
                'input_documents': [Document(metadata={}, page_content=shortened_doc)], 'output_text': shortened_doc
            }
            contexts.append(invoke_format) # doc을 그대로 context에 넣습니다
        else:
            if (len(shortened_doc) == 1000):
                print("Summarizer: Doc truncated to 1000 tokens, we have saved some GPU")
            else:
                print("Summarizer: Doc not truncated and added to summary, no GPU saved :(")
            split_docs.append(shortened_doc) # doc의 첫 n개 토큰을 요약할 문서 리스트에 넣습니다
    
    # 비동기적으로 요약 작업을 실행합니다
    summaries = await asyncio.gather(*[summarize_task(doc) for doc in split_docs])
    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()

    
    atexit.register(cleanup)
    
    return contexts + summaries  # 요약된 결과를 리스트 형태로 반환합니다.
