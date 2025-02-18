import search.serper as serper
import search.crawler as crawler
import concurrent.futures
import search.summarizer as summarizer
import asyncio
from langchain_community.llms import VLLM
from my_utils import timeit


def filter_link(search_results):
    """
    args: 
        search_results: 정리되지 않은 검색 결과 딕셔너리
    return:
        links_dict: 제목과 링크를 저장한 딕셔너리
    """
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
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_title = {executor.submit(fetch_data, title, link): title for title, link in filtered_links.items()}
        
        for future in concurrent.futures.as_completed(future_to_title):
            title, text = future.result()
            if text is not None:
                crawled_data[title] = text
    
    return crawled_data


@timeit
def search_pipeline(processed_query, llm, is_vllm):
    """
    처리된 query를 가지고 검색, 크롤링, 요약을 수행하는 파이프라인 함수
    args:
        processed_query: 처리된 query
        llm: 사용할 LLM
        is_vllm: vLLM 사용 여부
        
    return:
        summarized_results: 요약 결과
    """   

    print("\n\n==============Search api Result==============\n")
    search_results = serper.serper_search(processed_query) # api 호출
    filtered_links = filter_link(search_results)

    print(filtered_links)

    print("\n\n==============Crawling Result==============\n")
    final_results = crawl_links_parallel(filtered_links, crawler)
    # print(final_results)

    print("\n\n==============Summarization Result==============\n")
    summarized_results = asyncio.run(summarizer.summarize(list(final_results.values()), llm, is_vllm))
    return summarized_results

# ================================= Test ================================= #

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