import search.serper as serper
import search.crawler as crawler
import concurrent.futures


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
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_title = {executor.submit(fetch_data, title, link): title for title, link in filtered_links.items()}
        
        for future in concurrent.futures.as_completed(future_to_title):
            title, text = future.result()
            if text is not None:
                crawled_data[title] = text
    
    return crawled_data

def search_pipeline(processed_query):
    search_results = serper.serper_search(processed_query) # api 호출
    filtered_links = filter_link(search_results)
    print("\n\n==============Search api Result==============\n")
    print(filtered_links)
    final_results = crawl_links_parallel(filtered_links, crawler)
    print("\n\n==============Crawling Result==============\n")
    print(final_results)

    return final_results
