import serper as serper
import crawler as crawler
import concurrent.futures
import jin.query_search_merged_exp.search.summarizer_no_vllm as summarizer_no_vllm
import asyncio
from langchain_community.llms import VLLM
import time
import torch
import torch.distributed as dist
import os, re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from search_pipeline import search_pipeline
from langchain_core.output_parsers import StrOutputParser

# os.environ['MASTER_ADDR'] = 'localhost'  # 또는 실제 마스터 노드 IP 주소
# os.environ['MASTER_PORT'] = '12345'      # 임의의 사용하지 않는 포트


# query에서 processed_query를 아래 형식으로 받아왔음을 가정
# processed_query = [
#     {
#         "subquery": "날아다니는 파스타 괴물",
#         "routing": "web"
#     },
#     {
#         "subquery": "파스타 괴물 신화",
#         "routing": "web"
#     }
# ]




if __name__ == "__main__":
    query = "날아다니는 파스타 괴물 신화에 대해 알려줘"

    processed_query = [
    {
        "subquery": "날아다니는 파스타 괴물",
        "routing": "web"
    },
    {
        "subquery": "파스타 괴물 신화",
        "routing": "web"
    }
    ]
    
    llm = VLLM(model = "google/gemma-2-2b-it",
               trust_remote_code = True,
               max_new_tokens = 512,
               top_k = 10,
               top_p = 0.95,
               temperature = 0.9,
               gpu_memory_utilization = 0.8, # OOM 방지
               max_num_seqs = 8 # batch size 조정
               # tensor_parallel_size = 4 # for distributed inference
        
    )

    summarized_results = search_pipeline(processed_query, llm)
    
    contexts = []
    for result in summarized_results:
        contexts.append(result['output_text'])

    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=f"""아래 정보에 기반하여, 사용자의 질문에 답하세요.
        {contexts}
        사용자 질문: {query} """
    )           

    chat_prompt = prompt_template.format(context="\n".join(contexts), query=query)

    # runnable로 감싸기
    chat_runnable = RunnableLambda(lambda input: chat_prompt)
    
    chain = (
        chat_runnable
        | llm
        | StrOutputParser()
    )

    start_time = time.time()
    answer = chain.invoke(query)
    end_time = time.time()

    print(answer, f"Final Output Execution Time: {end_time-start_time}")