import search.serper as serper # removed search.
import search.crawler as crawler
import concurrent.futures
import search.summarizer as summarizer
import asyncio
from search.bad_links_list import bad_links
from search.weather import get_weather_forecast
from langchain_community.llms import VLLM
# from vllm import LLM
from my_utils import timeit
from urllib.parse import urlparse
from konlpy.tag import Okt
import threading

q = ""
w = ""
d = ""
weather_links = ["weawow.com", "korea247.kr", "windy.app"] # 도메인에 "weather" 안 들어간 날씨 사이트들
lock = threading.Lock()

def extract_place(subquery, flag):
    global q
    global w
    global d
    list_banned = ['날씨', '습도', '기온', '평균', '지역', '사용자', '지명', '비', '강수', '예보', '연간', '강수량', \
                   '내일', '현재', '모레', '글피', '어제', '오늘', '주간', '주말', '평일', '주중', '연휴', '아침', '저녁', \
                   '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일' \
                    '낮', '밤', '시간', '시', '분', '초', '날', '시기', '시점', '시간대', \
                    '이번', '지난', '저번', '다음']
    date_word = ['내일', '현재', '모레', '글피', '어제', '오늘', '주말', '평일', \
                 '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    if flag:
        with lock:
            q = subquery
            okt = Okt()
            noun = okt.nouns(subquery)
            # date_word에 해당하는 단어가 있으면 해당 단어 반환
            for date in date_word:
                if date in subquery:
                    d = date
                    break
            if d == "":
                d = '오늘'
            for word in noun:
                if word not in list_banned:
                    w = word
                    return word, d
            return None
    elif subquery == q:
        return w, '오늘'

def filter_link(search_results):
    # 어떤 제목의 링크를 타고 들어갔는지 기억하기 위해 dictionary 사용 (title을 key, link를 value로 저장)
    links = []
    for query in search_results:
        app = {query['searchParameters']['q'] + "+" + item['title']: item['link'] for item in query.get('organic', [])}
        links.append(app)
    links_dict = {}
    for link in links:
        links_dict.update(link)
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
def crawl_links_parallel(filtered_links, crawler, processed_query):
    global q
    crawled_data = {}
    weather_data = ""
    link_per_query = max(0, serper.k_num-3) #서브 쿼리 하나당 fetch 해올 url 개수 지정해주기
    
    def fetch_data(title, link):
        try:
            text = crawler.crawl(link)
            if text:  # valid한 텍스트가 들어온 경우
                return title, text  
        except Exception as e:
            print(f"Skipping {link} due to error: {e}")
        return None, None #에러가 난 경우 None 반환
    
    cnt = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for title, link in filtered_links.items():
            subquery = title.split("+")[0]
            flag = False
            if subquery != q: 
                with lock:
                    flag = True
                    q = subquery
                    cnt = 0

            if cnt >= link_per_query:
                continue
            
            # bad links는 크롤링에서 제외
            skip = False
            for item in bad_links:
                if item in link:
                    skip = True
                    break  # bad_links 중 하나라도 포함되면 건너뛰기

            if skip:
                continue  # bad_links에 포함된 경우 다음 링크로 넘어감

            parsed_url = urlparse(link)
            domain = str(parsed_url.netloc.lower())

            if "weather" not in domain and "kma.go.kr" not in domain and domain not in weather_links: # 날씨 관련 사이트가 아닌 경우
                future = executor.submit(fetch_data, title, link)
                title, text = future.result() 
                if text is not None:  # 에러가 난 페이지가 아닌 경우
                    print(f"Result: {title}, Length: {len(text)}")
                    crawled_data[title] = text
                    cnt += 1
            
            else: # 날씨 관련 사이트인 경우
                if weather_data == "":  # 날씨 데이터가 없는 경우
                    place_name, date = extract_place(subquery, flag)
                    text = get_weather_forecast(place_name, date)
                    if text is None:
                        future = executor.submit(fetch_data, title, link)
                        title, text = future.result() 
                        if text is not None:
                            print(f"Result: {title}, Length: {len(text)}")
                            crawled_data[title] = text
                            cnt += 1
                    else:
                        cnt += 1
                        weather_data += text

    return weather_data, crawled_data


@timeit
def search_pipeline(processed_query, llm, is_vllm):   

    print("\n\n==============Search api Result==============\n")
    search_results = serper.serper_search(processed_query) # api 호출
    filtered_links = filter_link(search_results) # api 답변에서 링크 추출

    print("\n\n==============Crawling Result==============\n")
    weather_result, final_results = crawl_links_parallel(filtered_links, crawler, processed_query) # 추출된 링크들에서 텍스트 추출

    print("\n\n==============Summarization Result==============\n")
    
    if hasattr(llm, "model_name"):
        model_name = llm.model_name
    else:
        model_name = llm.model
        
    summarized_results = asyncio.run(summarizer.summarize(list(final_results.values()), llm, is_vllm, model_name=model_name))
    summarized_results.append(weather_result)
    return summarized_results

# Test
if __name__ == "__main__":
    processed_query = [
    {
        "subquery": "김포 평균 기온",
        "routing": "web"
    },
    {
        "subquery": "부산 현재 날씨",
        "routing": "web"
    }
    ]
    
    llm = VLLM(
        model="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        trust_remote_code=True,
        max_new_tokens=128,
        # top_k=3,
        # top_p=0.1,
        temperature=0.9,
        do_sample=True,
        repitition_penalty=1.2,
        vllm_kwargs={"max_model_len": 10000}
    )

    # llm = VLLM(model = "google/gemma-2-2b-it",
    #            trust_remote_code = True,
    #            max_new_tokens = 128,
    #            top_k = 10,
    #            top_p = 0.95,
    #            temperature = 0.9,
    #            gpu_memory_utilization = 0.8, # OOM 방지
    #            max_num_seqs = 8, # batch size 조정
    #            #vllm_kwargs={"max_model_len": 5000}
    #            #tensor_parallel_size = 4 # for distributed inference   
    # )

    summarized_results = search_pipeline(processed_query, llm, 'true')
    print(summarized_results)
