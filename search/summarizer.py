import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from langchain.prompts import PromptTemplate
import asyncio
# from langchain.text_splitter import RecursiveCharacterTextSplitter # 추가
from my_utils import timeit
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# # 문서 분할기 (추가한 부분)
# def split_docs_by_token_count(docs, max_tokens):
#     # LLM의 최대 토큰 수에 맞춰 문서 분할
#     splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=50)
#     split_docs = []
#     for doc in docs:
#         split_docs.extend(splitter.split_text(doc))
#     return split_docs

async def summarize(docs, llm, is_vllm, max_tokens=2048, max_concurrent_tasks=4):
        
    """HuggingFace LLM으로 비동기적 요약을 실행"""
    if not docs:
        return {}


    prompt_template = """다음 텍스트를 읽고, 주요 아이디어와 핵심 내용을 파악하여, 간결하면서도 명확한 요약문을 작성해 주세요.

        요약문은 다음 항목들을 포함해야 합니다:
        1. 문서의 주요 주제 및 목적
        2. 핵심 내용과 중요한 세부사항
        3. 결론 및 전달하고자 하는 메시지

        외부 정보를 포함하지 않고 제공된 텍스트에만 전적으로 의존하세요.
        
        {input_documents}"""
    

    chat_prompt = PromptTemplate.from_template(prompt_template)
    
    # 요약 체인 로딩
    # chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False)
    chain = chat_prompt | llm | StrOutputParser()

    # 비동기 작업의 수를 제한하는 세마포어를 설정하여 동시에 실행되는 작업의 수를 max_concurrent_tasks(4개)로 제한합니다.
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # 문서 분할
    # split_docs = split_docs_by_token_count(docs, max_tokens) # 추가한 부분
    
    # 각 문서를 받아온 후 요약 작업을 수행하는 비동기 함수
    
    async def summarize_task(doc):
        async with semaphore:
            if is_vllm == 'false':  # vLLM 사용하지 않을 때만 비동기 실행
                response = ""  # 응답을 저장할 변수
                async for chunk in chain.astream({"input_documents": doc}):  # 스트리밍 데이터를 받음
                    response += chunk  # 부분 응답을 이어 붙임
                return response  # 최종 응답을 반환
            else:
                return await chain.ainvoke({"input_documents": doc})  # vLLM 사용 시 기존 방식 유지

            
    # 비동기적으로 요약 작업을 실행
    summarized_results = await asyncio.gather(*[summarize_task(doc) for doc in docs]) # 수정한 부분 (docs > split_docs)
    
    
    return summarized_results  # 요약된 결과를 리스트 형태로 반환합니다.
