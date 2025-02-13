import search.serper as serper
import search.crawler as crawler
import concurrent.futures
import search.summarizer as summarizer
import asyncio
from langchain_community.llms import VLLM
import torch.distributed as dist
import os
from my_utils import timeit

os.environ['MASTER_ADDR'] = 'localhost'  # 또는 실제 마스터 노드 IP 주소
os.environ['MASTER_PORT'] = '12345'      # 임의의 사용하지 않는 포트


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
@timeit
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


@timeit
def search_pipeline(processed_query, llm, is_vllm):   
    if os.environ.get('WORLD_SIZE', '1') != '1':
        dist.init_process_group(backend='nccl', world_size=1, rank=0)

    print("\n\n==============Search api Result==============\n")
    search_results = serper.serper_search(processed_query) # api 호출
    filtered_links = filter_link(search_results)
    if dist.is_initialized():
        dist.destroy_process_group()
    print(filtered_links)
    if dist.is_initialized():
        dist.destroy_process_group()

    print("\n\n==============Crawling Result==============\n")
    final_results = crawl_links_parallel(filtered_links, crawler)
    print(final_results)

    print("\n\n==============Summarization Result==============\n")
    summarized_results = asyncio.run(summarizer.summarize(list(final_results.values()), llm, is_vllm))
    if dist.is_initialized():
        dist.destroy_process_group()
    print(summarized_results)
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    contexts = []
    for result in summarized_results:
        contexts.append(result['output_text'])
    
    return contexts

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
    
    llm = VLLM(model = "google/gemma-2-2b-it",
               trust_remote_code = True,
               max_new_tokens = 128,
               top_k = 10,
               top_p = 0.95,
               temperature = 0.9,
               gpu_memory_utilization = 0.8, # OOM 방지
               max_num_seqs = 8 # batch size 조정
               # tensor_parallel_size = 4 # for distributed inference
        
    )

    summarized_results = search_pipeline(processed_query, llm, 'true')