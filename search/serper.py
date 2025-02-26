import asyncio
# 부모 경로 저장
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from search.config import SERPER_API_KEY
from my_utils import timeit

os.environ["SERPER_API_KEY"] = SERPER_API_KEY
k_num = 5 # k = 검색 개수, crawl_links_parallel 함수에서 최종적으로 가져오는 개수는 k - 2개임.

# Google Serper 비동기적 처리
serper_search_tool = GoogleSerperAPIWrapper(k=k_num)

async def async_serper_call(query, serper_search_tool):
    """비동기적으로 Serper Search API를 호출하는 함수"""
    return await serper_search_tool.aresults(query)

async def async_fetch_results(queries):
    """비동기적 검색을 위한 작업들 생성하는 함수"""
    tasks = [async_serper_call(query, serper_search_tool) for query in queries]
    results = await asyncio.gather(*tasks)  # 비동기적으로 search 작업 처리
    return results

@timeit
def serper_search(examples): 
    """__main__ 환경에서 실행가능하도록 asyncio.run() 처리"""
    queries = [example["subquery"] for example in examples if example["routing"] == "web"]    
    results = asyncio.run(async_fetch_results(queries)) 
    return results

# -----------------------------  테스트용 코드 ----------------------------- #

if __name__ == "__main__":
    examples = [
        {
            "subquery": "애플 아이폰 최신모델",
            "routing": "web" # to web search
        },
        {
            "subquery": "아이폰 15 프로 가격",
            "routing": "db"
        },
        {
            "subquery": "아이폰 15 프로 맥스 스펙",
            "routing": "db"
        },
        {
            "subquery": "갤럭시 S24 울트라",
            "routing": "web" # to web search
        },
        {
            "subquery": "갤럭시 S24 출시일",
            "routing": "db"
        }
        ]
    serper_search(examples)
