# import search.serper as serper
# import search.crawler as crawler
import serper as serper
import crawler as crawler
import concurrent.futures
# import search.summarizer as summarizer
import summarizer as summarizer
import asyncio
from langchain_community.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import torch
import torch.distributed as dist
import atexit
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

os.environ['MASTER_ADDR'] = 'localhost'  # 또는 실제 마스터 노드 IP 주소
os.environ['MASTER_PORT'] = '12345'      # 임의의 사용하지 않는 포트


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

def filter_link(search_results):
    # 어떤 제목의 링크를 타고 들어갔는지 기억하기 위해 dictionary 사용 (title을 key, link를 value로 저장)
    links_dict = {item['title']: item['link'] for search in search_results for item in search.get('organic', [])}
    return links_dict

def crawl_links(filtered_links, crawler):
    crawled_data = {}

    for title, link in filtered_links.items():
        text = crawler.crawl(link)  # 크롤링 실행
        crawled_data[title] = text  # 타이틀과 크롤링된 텍스트를 딕셔너리로 저장
    final_results = {k: v for k, v in crawled_data.items() if v is not None}
    
    return final_results

# 병렬 처리 함수
def crawl_links_parallel(filtered_links, crawler):
    crawled_data = {}
    
    def fetch_data(title, link):
        text = crawler.crawl(link)
        return title, text
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_title = {executor.submit(fetch_data, title, link): title for title, link in filtered_links.items()}
        
        for future in concurrent.futures.as_completed(future_to_title):
            title, text = future.result()
            if text is not None:
                crawled_data[title] = text
    
    return crawled_data


def search_pipeline(processed_query, llm):   
    if os.environ.get('WORLD_SIZE', '1') != '1':
        dist.init_process_group(backend='nccl', world_size=1, rank=0) 
    search_results = serper.serper_search(processed_query) # api 호출
    filtered_links = filter_link(search_results)
    if dist.is_initialized():
        dist.destroy_process_group()
    print("\n\n==============Search api Result==============\n")
    print(filtered_links)
    if dist.is_initialized():
        dist.destroy_process_group()
    final_results = crawl_links_parallel(filtered_links, crawler)
    print("\n\n==============Crawling Result==============\n")
    print(final_results)
    summarized_results = asyncio.run(summarizer.summarize(list(final_results.values()), llm))
    if dist.is_initialized():
        dist.destroy_process_group()
    print("\n\n==============Summarization Result==============\n")
    print(summarized_results)
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return summarized_results

# Test
if __name__ == "__main__":
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
    
    model_name = "google/gemma-2-2b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                            torch_dtype=torch.bfloat16, 
                                             device_map="auto", # accelerator 이미 사용하고 있음.
                                             use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # 원하는 최대 토큰 길이 설정
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    # Langchain에 넣어주기 위해서 pipeline으로 감싸기
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    

    start_time = time.time()
    summarized_results = search_pipeline(processed_query, llm)
    end_time = time.time()
    
    print(summarized_results, f"Search ~ Summarization Execution Time: {end_time-start_time}")