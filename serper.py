# %pip install -qU langchain-google-community
# %pip install -qU langchain-community

import asyncio
import os
import pprint

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool

os.environ["SERPER_API_KEY"] = "506067cc3b3c6ee2a353392fa25206f87e8431d6"

# Google Serper 비동기적 처리
serper_search_tool = GoogleSerperAPIWrapper(k=5) # k = 검색 개수

async def async_serper_call(query, serper_search_tool):
    """비동기적으로 Serper Search API를 호출하는 함수"""
    return await serper_search_tool.aresults(query)

async def async_fetch_results(queries):
    # 비동기적 검색을 위한 작업들 생성
    tasks = [async_serper_call(query, serper_search_tool) for query in queries]
    results = await asyncio.gather(*tasks)  # 모든 작업을 비동기적으로 처리
    return results

def serper_search(examples):
    queries = [example["subquery"] for example in examples if example["routing"] == "web"]    
    results = asyncio.run(async_fetch_results(queries)) 
    # if __name__ == "__main__": 에서는 await을 직접 쓸 수 없어서 asyncio.run()을 사용해야 함
    
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        pprint.pprint(result)
        print()

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