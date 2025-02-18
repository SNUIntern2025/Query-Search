from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time
import asyncio
import torch.distributed as dist
import atexit

async def summarize(docs, llm, max_tokens=1000, max_concurrent_tasks=5):
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

    # 각 문서를 받아온 후 요약 작업을 수행하는 비동기 함수입니다
    # vllm과의 충돌 문제 때문에, 우선 ainvoke 제외하여 실행하였습니다.
    async def summarize_task(doc):
        async with semaphore:
            return await chain.ainvoke({"input_documents": [Document(page_content=doc)]})
            #return chain.invoke({"input_documents": [Document(page_content=doc)]})
    # 비동기적으로 요약 작업을 실행합니다
    summaries = await asyncio.gather(*[summarize_task(doc) for doc in docs])
    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()
    
    atexit.register(cleanup)
    
    return summaries  # 요약된 결과를 리스트 형태로 반환합니다.
