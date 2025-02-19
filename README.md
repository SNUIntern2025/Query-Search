# DAG with AI Agents

**Table of Contents**

- [main.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)

Query

- [query_pipeline.py](https://www.notion.so/2025-Winter-SNU-NLP-16e245846c0a80109ed3cb2dba6f3ba7?pvs=21)
- [query_with_gemma2.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)
- [parallel.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)

Search

- [search_pipeline.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)
- [serper.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)
- [crawler.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)
- [summarizer.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)
- [config.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)

Final Output

- [final_output.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)

기타

- [my_utils.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)
- [app.py](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)
- [requirements.txt](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)

[Unsolved Issues](https://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=21)

# 실행 방법

## `main.py`

- 이 파일을 실행함으로써 전체 agent ai가 동작합니다. 두 가지 방식으로 모델을 로드할 수 있습니다.
1. **OpenAI Compatible API를 이용하여 API를 가져오듯이 모델을 호출하는 경우**
    - **cli command**로 아래의 코드를 넣어 모델을 먼저 서빙해주세요.
        - 기본 옵션으로 데이터타입은 자동으로 들어가도록, 텐서 병렬처리는 4개의 GPU가 나누어 하도록 설정해두었습니다.
    
    ```bash
     nohup vllm serve snunlp/bigdata_gemma2_9b_dora --dtype auto --tensor-parallel-size 4 --api-key token-snuintern2025 & 
    ```
    
    - command에 추가 적용할 수 있는 옵션과 관련하여서는 [이 문서](https://www.notion.so/02-18-19d245846c0a808c9784e0fca800dad8?pvs=21)를 참조해주세요.
        - `--enable-prefix-caching` 는 넣는 것을 추천드립니다.
        - `--enable-chunked-prefill`, `--block-size (2, 4, 8, 16, 32까지 지정 가능)`옵션도 나름대로 도움이 될 수도 있을 것 같습니다.
    - 현재 local 호스트, 포트 8000으로 연결해두었습니다. 필요에 따라 변경하시면 됩니다.
2. **로컬 환경에서 직접 모델을 이용하는 경우**
    - 아래의 코드를 통해 실행 가능
        
        ```bash
        python main.py --vllm=true
        ```
        
    
    > **인자 설명**
    - `vllm` : vLLM을 사용할지 말지 결정하는 인자. “true”를 주면 vLLM으로 wrapping 된 채로 실행됩니다.
    > 
    - 실행 후
        
        ```bash
        입력> 
        ```
        
        창이 뜨면 사용자 쿼리를 입력해주시면 됩니다.
        

### 함수 설명

- **`load_model(MODEL_NAME)`**
    - vLLM을 사용하지 않고 모델을 불러옵니다.
    
    > **args**
        MODEL_NAME (str): 모델 이름. huggingface 모델명이나, 로컬 모델 체크포인트 경로를 입력
    > 
    
    > **return**
        llm (HuggingFacePipeline) : 로딩이 된 모델. huggingface pipeline에 래핑되어 리턴됨
    > 

- **`load_vllm_1(MODEL_NAME)`**
    - vLLMOpenAI()를 이용하는 경우로, vLLM을 백엔드 서버로 띄워두고 API를 호출하듯 모델을 사용하는 방식입니다.
    - 이후 `if __name__ == "__main__":` 아래의 `load_func` 에서 이 함수 혹은 `load_vllm_2`를 선택하여 수정하시면 됩니다.
        - 인자와 반환값은 `load_vllm2` 함수와 같습니다.
        
- **`load_vllm_2(MODEL_NAME)`**
    - vLLM을 사용하여 모델을 불러옵니다.
    
    > **args**
    - MODEL_NAME (str): 모델 이름. huggingface 모델명이나, 로컬 모델 체크포인트 경로를 입력
    > 
    
    > **return**
    - llm (VLLM) : 로딩이 된 모델. VLLM에 래핑되어 리턴됨
    > 

# Query Part

- 경로: `query/` 안의 모든 파일들

## `query_pipeline.py`

- 서브쿼리 분리 및 쿼리 라우팅을 실행하는 함수가 있는 파일
- 해당 코드를 통해 쿼리 파이프라인만 실행해 볼 수 있습니다.
    
    ```python
    python query_pipeline.py --vllm=true
    ```
    

### **함수 설명**

- 🚩 **`query_pipeline(query, llm, is_vllm)`**
    - 쿼리 파트 (서브 쿼리 분해, 쿼리 라우팅) 전체 파이프라인을 수행하는 함수
    - 아래와 같은 로직으로 구성되어 있습니다.
    
    ```python
    서브쿼리 = 서브쿼리 라우팅 함수(사용자 입력)
    서브쿼리 라우팅 결과 = 쿼리 라우팅 함수(서브쿼리)
    return 서브쿼리, 서브쿼리 라우팅 결과
    ```
    
    > **args**
    > 
    > 
    >     query (str): 사용자 입력 쿼리
    > 
    >     llm (LangChain.VLLM or huggingface.pipeline) 사용할 llm
    > 
    >     is_vllm (str): vllm 사용 여부 ("true" | "false")
    > 
    
    > **return**
    > 
    > 
    >     subqueries (list[str]), 분해된 서브쿼리
    > 
    >     final_processed_query (list[Dict]): 서브쿼리 라우팅 결과
    > 
    - 예시
        
        ```python
        query = "피크민이 뭐야?"
        query_pipeline(query, llm, args.vllm)
        
        >> ["피크민의 정의, 피크민의 특징"],
           [{"subquery": "피크민의 정의", "routing": "web"},
            {"subquery": "피크민의 특징", "routing": "web"}]
        ```
        

- **`load_model` , `load_vllm`**
    - `main.py` 에 있는 동명의 함수와 기능이 동일합니다. 이 함수들 대신 main.py에 있는 함수만 쓰입니다. 자세한 사항은 [main.py](http://main.pyhttps://www.notion.so/2-19-19d245846c0a805e8589ed59286cec6b?pvs=4#19d245846c0a800eae2bec4168d7d2c9) 참조

## `query_with_gemma2.py`

- 서브쿼리 분리를 실행하는 함수

### 인자 설명

- special_tokens (Dict)
    - 프롬프트 생성을 위해, 모델마다 통용되는 special token을 저장해놓은 함수.
    - system, user, assistent, example, end token을 저장

### 함수 설명

- 🚩 **`get_sub_queries(query, llm)`**
    - 사용자 입력을 받아, 하위 쿼리로 나누어 반환하는 함수
    
    > **args**
            query (str) : 사용자 입력
            llm : 생성할 때 사용할 llm pipeline
    > 
    
    > **return**
            sub_queries (List[str]) : 하위 쿼리
    > 
    - 예시
        
        ```python
        get_sub_queries("피크민이 뭐야?", llm)
        
        >> ["피크민의 정의", "피크민의 특징"]
        ```
        
- **`load_prompt(system_prompt, model_name, fewshot_ex)`**
    - LangChain에 사용할 수 있는 프롬프트를 생성하는 함수
    
    > **args**
            system_prompt (str): 시스템 프롬프트
            model_name (str): 모델명
            fewshot_ex (list): few-shot 학습 데이터
    > 
    
    > **return:**
            chat_prompt (PromptTemplate) : 랭체인에 들어갈 프롬프트
    > 
    - 시스템 프롬프트는 `query/subquerying_prompt.py` 안에 포함되어 있습니다.
        - 프롬프트 예시
            
            ```python
            SYSTEM_GEMMA = """<start_of_turn>system 당신은 들어오는 복잡한 질문들을 단순한 하위 작업으로 나누는 데에 능통한 전문가입니다.
            1. 주어진 질문을 순차적으로 세부 작업으로 나누세요.
            2. 각 하위 쿼리는 하나의 작업만으로 이루어져야 합니다. 절대 여러개의 작업을 포함하지 마세요. (예: 동시에 두 개 이상의 정보를 조사하는 작업 안 됩니다. 가격대 등도 마찬가지로 각각 따로 질문을 던지도록 하세요.)
            3. 3개 이하의 하위 쿼리를 생성하세요. 주어진 질문이 여러 문장인 경우 더 많이 생성해도 됩니다. 최대 8개까지만 생성하세요.
            4. 답변은 아래와 같이 JSON 형식으로 해야 합니다.
            <start_of_turn>model
            {{
                "response": ["하위 쿼리 1", "하위 쿼리 2", ...],
            }}
            <end_of_turn>
            5. 답변에는 JSON 형식 이외의 다른 내용을 포함하지 마세요. (```json``` 태그로도 작성하지 마세요.)
            6. 오늘 날짜를 반영하여 결과를 생성하세요.<end_of_turn>"""
            ```
            
    - few shot_ex에 들어갈 예시 데이터는 `few_shot.py`에 들어가 있습니다.
        - 데이터 예시
            
            ```python
            examples_final = [
                {
                    "input": "딸기랑 바나나는 각각 어느 계절에 자라? 둘중에 내일 무슨 과일 먹을지 추천해줘.",
                    "output": """{{
                        "response": ["딸기가 자라는 계절", "바나나가 자라는 계절", "딸기와 바나나 중 먹을 것 추천"]
                        }}"""
                }, ...]
            ```
            

## `parallel.py`

- 쿼리 라우팅을 진행하는 함수가 있는 파일

### 함수 설명

- 🚩 **`prompt_routing(subqueries: List[str], llm, is_vllm)`**
    - subquery를 받아, LLM prompting을 통해 routing을 병렬로 실행하는 함수
    
    > **args**
        subqueries (List[str]): 서브쿼리 분해 함수에서 최종적으로 리턴한, 서브쿼리가 들어있는 리스트
        llm: 생성할 때 사용할 llm pipeline
        is_vllm (str): vllm 사용 여부. 멀티쓰레딩으로 코드가 구현되어 있을 때는 사용했으나 현재는 사용하지 않습니다.
    > 
    
    > **return**
        all_responses (List[Dict]): 서브쿼리 별 라우팅 결과가 들어있는 리스트
    > 
    - 예시
        
        ```python
        prompt_routing(["피크민의 정의", "피크민의 특징"], llm, is_vllm)
        
        >> [{"subquery": "피크민의 정의",
            "routing": "web",
            "reasoning": "피크민의 정의를 알려면 외부 정보가 필요합니다."},
            {"subquery": "피크민의 특징",
            "routing": "web",
            "reasoning": "피크민의 특징을 알려면 외부 정보가 필요합니다."}]
        ```
        
    
- **`load_prompt(system_prompt, model_name)`**
    - LangChain에 사용할 수 있는 프롬프트를 생성하는 함수 (for gemma2)
    
    > **args**
            system_prompt (str): 시스템 프롬프트
            model_name (str): 모델명
    > 
    
    > **return:**
            chat_prompt (PromptTemplate) : 랭체인에 들어갈 프롬프트
    > 
    
- **`load_prompt_exaone(system_prompt, model_name, fewshot_ex)`**
    - LangChain에 사용할 수 있는 프롬프트를 생성하는 함수 (for EXAONE)
    
    > **args**
            system_prompt (str): 시스템 프롬프트
            model_name (str): 모델명
            fewshot_ex (list): few-shot 학습 데이터
    > 
    
    > **return:**
            chat_prompt (PromptTemplate) : 랭체인에 들어갈 프롬프트
    > 
    - 시스템 프롬프트는 query/routing_prompts.py 파일 안에 포함되어 있습니다.
        - 프롬프트 예시
            
            ```python
            PARALLEL_GEMMA= """
            주어진 쿼리에 대답하기 위해, 인터넷 정보 검색이 필요한지 판단해야 한다.
            
            검색 실행 여부에 대한 판단 기준은 다음과 같다:
            1. 질문 유형: 질문이 사고를 요구하는가, 정보를 요구하는가?
            2. 맥락: 질문에 포함된 정보만으로 답변을 구성할 수 있는가?
            
            인터넷 검색 간 판단 기준은 다음과 같다:
            1. 시의성: 주제가 최신 정보와 관련이 있는가? 최신 정보가 답변의 질을 향상시키는가?
            2. 정밀성: 답변을 구성하기 위해 구체적이고 정확한 정보가 필요한가?
            3. 전문성: 질문의 주제가 전문 지식이나 첨단 연구와 관련이 있는가?
            4. 신뢰성: 외부 자료를 통한 교차검증이 답변의 질을 향상시키는가?
            
            답변은 오직 아래 예시와 같은 JSON 형식으로 해야 하며, 다른 코드나 내용을 포함해서는 안 된다.
            routing은 인터넷 검색의 필요 여부를 나타낸다. (필요한 경우 "web", 아닐 경우 "none")
            {{"subquery": "대전 주요 관광지 추천", "routing": "web", "reasoning": "대전의 주요 관광지는 시의성이 크고 신뢰성이 중요하므로, 외부 정보를 필요로 합니다."}}
            {{"subquery": "지난 주 있었던 지진에 대해 알려줘", "routing": "web", "reasoning": "지난 주 있었던 지진은 시의성이 크므로, 외부 정보를 필요로 합니다."}},
            {{"subquery": "2x + 4 = 0에서 x는?", "routing": "none", "reasoning": "일차방정식의 풀이는 외부 정보를 필요로 하지 않습니다."}}
            """
            ```
            
        
- **`post_process_result(model_name, text)`**
    - 모델이 반환한 텍스트를 json형식으로 만들기 적합하도록 후처리 하는 함수
    
    > **args**
       model_name (str): 모델명
       text (str): 후처리 할 텍스트
    > 
    
    > **return**
        json_result (Dict): json으로 변환된 결과
    > 
    
- **`process_single_query(query, chain, model_name)`**
    - 멀티쓰레딩 작업을 삭제하고 batch 단위로 쿼리를 처리하는 작업으로 변경. 따라서 이 함수는 현재 사용하지 않습니다.

# Search Part (자세한 설명은 [채은님 노션](https://www.notion.so/197245846c0a80bcad60d3c10dfd2812?pvs=21)에)

- 경로: `Query+Search_vllm/search`  안의 모든 파일들

## `search_pipeline.py`

- 서브쿼리를 받아 검색을 수행하고, 크롤링을 해 오고, 이를 요약하는 함수가 들어 있는 파일
- 아래 코드를 통해 서치 파이프라인만 실행해볼 수 있습니다.
    
    ```jsx
    python search_pipeline.py
    ```
    
    - vllm이 디폴트로 적용됩니다.
    - 스파게티 괴물 신화 관련 내용이 디폴트 값으로 들어갑니다.
        - 이 입력이 들어갑니다.
            
            ```python
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
            ```
            

### 함수 설명

- 🚩 **`search_pipeline(processed_query, llm, is_vllm)`**
    - 서브쿼리를 받아 검색을 수행 → 크롤링 → 크롤링 결과 요약을 수행하는 함수
    
    > **args**
        processed_query (List[Dict]): 서브쿼리와 라우팅 결과가 딕셔너리로 들어 있는 리스트
        llm: 생성할 때 사용할 llm pipeline
        is_vllm: vllm 사용 여부. 멀티쓰레딩을 사용할 때 썼던 인자. 이제는 사용하지 않습니다.
    > 
    
    > **return**
       summarized_results (List[str]): 검색 결과별로 이를 요약한 텍스트가 들어 있는 리스트
    > 
    - 입력 예시
        
        ```python
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
        ```
        
    - 출력 예시
        
        ```python
        search_pipeline(processed_query, llm, is_vllm)
        
        >> ["날아다니는 파스타 괴물은 ...", "파스타 괴물 신화는 ...", ...]
        ```
        
    
- **`filter_link(search_results)`**
    
    > **args**
        search_results (Dict[Dict]): 각 Dict는 각 서브쿼리에 대해 서치해온 링크 및 사이트 제목 정보를 담고 있음
    > 
    
    > **return**
       filtered_links (List[Dict]): 그 중 링크에 대한 정보만 추출하여 다시 List[Dict]로 리턴
    > 
    
- **`crawl_links_parallel(filtered_links, crawler), crawl_links(filtered_links, crawler)`**
    - serper api가 반환해준 url을 이용하여 크롤링을 실행하는 함수
    - `search/crawler.py`에 있는 `crawl` 함수를 사용하여 크롤링 실행
    - `crawl_links_parallel` 은 멀티쓰레딩 함수, `crawl_links` 은 해당 멀티쓰레딩 작업을 수행하기 위한 내부 함수
    
    > **args**
        filtered_links (Dict): 검색 결과 문서 제목과 url이 딕셔너리 형태로 담겨 있는 딕셔너리
        crawler (module): 크롤링 모듈. `search/crawler.py` 파일
    > 
    
    > **result**
        final_results (Dict): 타이틀과 크롤링된 텍스트가 담겨 있는 딕셔너리
    > 
    - 예시
        
        ```python
        filtered_links = {"갤럭시 s25 정보 총정리": "https://url1", ...}
        crawl_links_parallel(filtered_links, crawler)
        
        >> {"갤럭시 s25 정보 총정리": "이번 달 공개된 갤럭시 s25 출시 스펙, 디자인, 가격까지 총정리...", ...}
        ```
        

## `serper.py`

- google serper api를 이용하여 검색을 수행하는 함수
- 비동기적으로 동작

### 함수 설명

- **`GoogleSerperAPIWrapper()`**
    - 서치 툴을 Wrapper로 감싸주기
    
    > **args**
    > 
    > 
    > `k_num` : 검색하여 반환할 웹사이트 개수 설정
    > 
    
    > **return**
    > 
    > 
    > `GoogleSerperAPIWrapper` 객체
    > 

- **🚩 `def serper_search(examples)`**
    - 여러 개의 검색어를 비동기적으로 Google Serper API를 통해 검색하고, `__main__` 환경에서 실행가능하도록 `asyncio.run()` 처리하는 함수
    
    > **args**
    > 
    > 
    > `examples` (list[dict]): 검색할 서브쿼리와 라우팅 내용을 담은 list[dict]
    > 
    > - `"subquery"` (str): 검색할 하위 쿼리
    > - `"routing"` (str): `"web"`인 경우 웹 검색을 수행
    
    > **return**
    > 
    > 
    > `results` (Dict[dict]) : GoogleSerper API가 긁어온 딕셔너리.
    > 
    > - 자세한 구조
    >     
    >     ```python
    >     {'answerBox': {'link': ..., 'snippet':..., 'title': ...},
    >     	'credits': 1,
    >     	'organic': 
    >     		[{'link': ..., 'snippet':..., 'title': ...},
    >     		 {'link': ..., 'snippet':..., 'title': ...},
    >     		...],
    >     	'peopleAlsoAsk':
    >     		[{'link': ..., 'question': ..., 'snippet': ..., 'title': ...}, 
    >     		 {'link': ..., 'question': ..., 'snippet': ..., 'title': ...} ...],
    >     	'relatedSearches': 
    >     		[{'query': ...}, {'query': ... }, ...],
    >     	'searchParameters': {'engine': 'google', 'gl': 'us',                      'hl': 'en', 'num': 10, 'q': '(주: 쿼리내용)', 'type': 'search'}
    >     		
    >     ```
    >     
    >     - searchParameters
    >         - engine(구글), gl(나라), hl(언어), num(검색 문서 수?), q(쿼리), type(**서치 옵션 - image, map 등 지정 가능**)

- **`async def async_serper_call(query, serper_search_tool)`**
    - 특정 검색어에 대해 비동기적으로 Serper Search API를 호출하는 함수
    
    > **args**
    > 
    > 
    > `query` (str): examples에서 “subquery”의 내용에 해당하는 것들만 모아놓은 리스트 안에서 뽑혀 나온 원소.
    > 
    > `serper_search_tool` (GoogleSerperAPIWrapper) : 검색을 수행할 API wrapper 객체
    > 
    
    > **return**
    > 
    > 
    > `await serper_search_tool.aresults(query)` (Dict[dict])
    > 
    > - 자세한 구조는 위 🚩 `serper_search` 함수 참조
    > - `await` : `async` 함수(비동기) 내에서 사용, 다른 작업이 동시에 진행될 때 현재 작업을 잠시 멈출 수 있음을 의미
    > - `aresult`: 비동기적인 방식으로 검색 API 요청을 보내는 함수

- **`async def async_fetch_results(queries)`**
    - 비동기적 검색을 위한 작업들을 생성하고 실행하는 함수.
    - `asyncio.gather()`를 사용하여 모든 검색을 병렬로 처리함
    
    > **args**
    > 
    > 
    > `queries` (list[str]) : 검색할 서브쿼리들의 리스트
    > 
    
    > **return**
    > 
    > 
    > `results` (Dict[dict]): GoogleSerper API가 긁어온 딕셔너리. 
    > 
    > - 자세한 구조는 위 🚩 `serper_search` 함수 참조

수정하신 내용이 있으시면 보완 부탁드립니다 !

## `crawler.py`

- 크롤링을 담당하는 모듈입니다. 아래의 함수들로 구성되어 있습니다.
    - 구조를 **알고 있는** 몇몇 웹사이트들에서 주요 내용을 긁어올 수 있는 핸들러 함수들
    - 구조를 파악하지 못한 웹사이트들을 처리하는 **fallback logic**으로서 주요 내용을 가져오는 함수 (trafilatura 라이브러리 이용)
    - 실제로 크롤링을 하는 함수
- 비동기적으로 동작
- 확장성을 고려하여, 사이트 패턴과 핸들러를 딕셔너리로 관리하고 있습니다.

### 함수 설명

- **🚩 `crawl(url)`**
    - Serper API를 통해 받아온 웹사이트가 우리가 알고있는 웹사이트들에  해당한다면
        - 1) 해당 웹사이트 핸들러를 이용하여 크롤링을 수행하고
        - 2) 그렇지 않으면 fallback logic으로 넘어가 크롤링을 수행하는 함수.
        
        > **args**
        > 
        > 
        > `url` (str): serper API를 통해 받아온 웹사이트 url
        > 
        
        > **return**
        `result` ****(str) : 크롤링을 한 웹사이트에서 끌어온 주요 내용
        > 
    
- **`dispatch_known_site(url)`**
    - URL이 알려진 사이트 패턴과 일치하는지 확인하고 해당되는 웹사이트의 핸들러를 호출하는 함수
        
        > **args**
        > 
        > 
        > `url` (str): serper API를 통해 받아온 웹사이트 url
        > 
        
        > **return**
        `handler(url)` ****(func) : 이미 그 구조를 알고 있는 웹사이트들을 관리하는 함수
        > 
        
    - **`KNOWN_SITE_HANDLERS`** : ****현재까지 main content의 위치가 파악된 사이트 모음을 관리하는 딕셔너리
        - 일부 예시:
        
        ```python
        {r"ko\.wikipedia\.org": handle_kor_wikipedia, 
        	r"n\.news\.naver\.com/article" : handle_naver_news, ...}
        ```
        

- **`fallback_extraction(url)`**
    - trafilatura 라이브러리를 사용하여 main content를 추출하는 함수
    - 우리가 그 구조를 잘 알지 못하는 경우 주요 내용을 추출해옴 (fallback logic)
        
        > **args**
        > 
        > 
        > `url` (str): serper API를 통해 받아온 웹사이트 url
        > 
        
        > **return**
        `parsed_result[’text']` ****(str) : trafilatura 라이브러리가 가져온 웹사이트의 주요 본문 내용
        > 

- 알고있는 웹사이트들에서 주요 내용을 뽑아오는 함수들
    - `handle_arxiv(url)` , `handle_kor_wikipedia(url)`, `handle_dbpia(url)`, `handle_kyobo(url)` , `handle_SOF(url)`, `handle_velog(url)`, `handle_tistory(url)`, `handle_daum_news(url)`, `handle_naver_news(url)`, `handle_naver_blog(url)`
    - 기본적인 구조는 비슷합니다. 웹사이트에 정상적으로 접속이 된 경우에 주요 내용을 포함하고 있는 html 태그를 찾아 그 내용을 반환하는 식입니다.
    
    > **args**
    > 
    > 
    > `url` (str): serper API를 통해 받아온 웹사이트 url
    > 
    
    > **return**
    `handler 함수에 따라 변수명 다름` ****(str) : 크롤링을 한 웹사이트에서 끌어온 주요 내용
    > 
    - arxiv의 경우 [export.arxiv.org](http://export.arxiv.org/) API를 사용하여 논문의 초록을 가져오도록 하는 `get_arxiv_abstract(arxiv_id)` 함수를 함께 이용하도록 되어 있습니다.

## `summarizer.py`

- 가져온 인터넷 정보들을 최종 답변 작성 이전에 요약하는 함수
- 한 인터넷 정보가 한 llm의 input으로 들어감 (쿼리 라우팅과 비슷)
- 비동기로 실행

### 함수 설명

- **🚩`summarize(docs, llm, is_vllm, max_tokens, max_concurrent_tasks, model_name)`**
    - 요약할 문서들을 받아 요약한 결과를 반환하는 비동기 함수
    - 크롤링 해 온 문서의 토큰 개수를 센 후 그 개수를 기준으로 그대로 사용할지, 요약할지 결정
    - vllm을 사용하지 않을 경우 ainvoke로 여러 문서를 llm으로 요약하고, 사용할 경우 batch로 여러 문서를 요약함
    
    > **args**
    > 
    > 
    >     docs (List[str]): 각 str는 태그를 땐 웹 문서를 나타냄
    > 
    >     llm: 생성할 때 사용할 llm
    > 
    >     is_vllm: vllm 사용 여부
    > 
    >     max_tokens: Unused
    > 
    >     max_concurrent_tasks[int]: 동시에 실행할 최대 요약 작업의 개수
    > 
    >     model_name[str]: 요약 모델의 이름
    > 
    
    > **return**
        contexts + summaries (List): final_output에 사용할 최종 컨텍스트. 요약 내용을 담은 문자열들이 리스트로 반환됨
    > 
    
- **`truncate(doc, count, model_name)`**
    - 문서의 앞부분을 특정 개수의 토큰만큼 추출하는 함수
    
    > **args**
    > 
    > 
    >     doc (str): 태그를 땐 웹 문서
    > 
    >     count (int): 추출할 토큰의 개수
    > 
    >     model_name(str): 사용할 tokenizer의 모델명
    > 
    
    > **return**
        tokenizer.decode(tokens) (str): 짧아진 문서
    > 
    
- **`summarize_task(doc)`**
    - 개별 문서를 요약하는 비동기 함수.
    - vllm을 사용하지 않을 때만 사용
    
    > **args**
    > 
    > 
    >     doc (str): 태그를 땐 웹 문서
    > 
    
    > **return**
        result (str): 요약된 문서
    > 

## `config.py`

- Google Serper API key를 담은 파일입니다.
- 채은과 혜진의 key가 들어있는데, 직접 아래 링크에서 key를 발급받으셔서 추가하셔도 되고, 테스트 과정에서는 저희 key를 사용하셔도 됩니다. 만약 사용량 제한이 걸리면 결제가 필요합니다.
    - https://serper.dev/?utm_term=google%20search%20api&gad_source=1&gclid=CjwKCAiAn9a9BhBtEiwAbKg6fk30ZmMhb4MQcdcxZgdE-fqyNDhCq6mmAPt7HCiRbab9q_P2b4ZSFxoCu54QAvD_BwE
    - 가격 정보
        - 최초 2500건 무료, credit 구매 후 **6개월 이내 사용해야 함.**
        - 2500건 이후:
            - 50,000 credit: $50 ($1/1k)
            - 500,000 credit: $375 ($0.75/1k)
            - 2,500,000 credit: $1250 ($0.50/1k)
            - 12,500,000건: $3750 ($0.30/1k)

# Final output part

## `final_output.py`

- 가져온 모든 인터넷 정보들을 통합하여 모델이 최종 답변을 내는 함수가 있는 파일
- 해당 코드를 실행하여 해당 코드만 실행해볼 수 있습니다.
    - vllm이 디폴트로 적용됨
    
    ```jsx
    python final_output.py
    ```
    

### 함수 설명

- **🚩 `final_output(query, contexts, llm)`**
    - query와 context를 이용하여 prompt를 만든 뒤,
    - chain으로 prompt와 llm, StrOutputParser()가 연결되어서 최종 답변을 생성해줍니다.
    
    > **args**
    > 
    > 
    >    query (str): 사용자가 넣은 입력 쿼리
    > 
    >    contexts (List(str)): 검색 후 크롤링 된 문서 요약 과정을 거쳐 받아온 context. 
    > 
    >    llm : context를 활용하여 응답을 생성하는 모델
    > 
    
    > **return**
        answer (str) : 모델이 생성한 최종 응답
    > 
    

# 기타 함수

## `my_utils.py`

- 특정 함수의 소요 시간을 측정하는 데코레이터 `timeit` 존재
- 해당 데코레이터를 함수 위에 붙여주면, 해당 함수가 동작하는 데 걸린 총 시간을 측정하여 출력 해 줍니다.
    - 사용 예시
        
        ```python
        from my_utils import timeit
        
        @timeit
        def query_pipeline(query, model_name, llm, is_vllm):
        ...
        ```
        
        - 결과
        
        ```jsx
        query_pipeline 소요 시간: ...
        ```
        

## `app.py`

- 데모 시연을 위해 gradio를 실행하는 함수

```markdown
python app.py
```

- 서버로 연결되어 챗봇을 실행할 수 있습니다.
- 실행 화면
    
    ![image.png](image.png)
    

## `requirements.txt`

- 실행에 필요한 설치 파일들이 적혀있는 텍스트 파일
- 아래 함수를 이용하여 설치해주시면 됩니다.

```bash
pip install -r requirements.txt
```

## 🚑 Unsolved Issues

- 아직 고민 중 / 해결 시도 중인 문제들이 있어 공유드립니다. documentation을 드린 이후에도 저희가 보완할 예정입니다!
- **치명적인 이슈**
    - ⚠️ query 처리 결과 json 오류로 실행이 중단되는 문제
        - 문제: subquery 분해 및 query routing을 LLM이 수행하다보니 적절하지 않은 형식을 뱉을 때 아래와 같은 오류가 발생합니다.
        - 해결: 정규식을 이용하여 보완하는 방안
        
        ```bash
            raise JSONDecodeError("Expecting value", s, err.value) from None
        json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
        ```
        
    - ⚠️ 모델 자체의 max length가 짧은 문제
        - 최종 답변이 잘려서 나오거나, 아예 생성이 안 됨.
        - context가 약 22000 토큰으로 길게 주어지면 max length인 8192보다 커져 에러 발생
        - 해결방안 : summarizer.py 보완 예정
            - 답변을 위해 남겨둘 길이가 최소한 얼마나 되어야 할지에 대한 합의가 필요함.
        
        ```bash
        입력 >  신효필 교수님과 그 컴퓨터언어학 연구실에 대해 알려줘
        ```
        
        ```bash
        openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': "This model's maximum context length is 8192 tokens. However, you requested 27022 tokens (22926 in the messages, 4096 in the completion). Please reduce the length of the messages or completion.", 'type': 'BadRequestError', 'param': None, 'code': 400}
        ```
        
- **치명적이지 않은 이슈** (성능 저하/저작권 이슈)
    - ⚠️ 나무위키 전용 크롤링 메서드가 없어서, 성능이 떨어지는 fallback logic이 실행됨
        - **예시:** 전체 문서가 아닌 3.2 단락만 크롤링됨
            
            ```python
            URL: https://namu.wiki/w/IVE
            4세대 걸그룹답게
            걸 크러시
            (girl crush) 콘셉트를 기반으로 데뷔를 했으나, 흔히 아이돌판에서 걸크러시의 대표격으로 인식되는 센 언니 기믹
            [3]
            대신에 ‘
            소녀
            ’라는 정체성을 유지하면서
            자기애를 드러내는 아름다운 소녀 이미지
            를 구축해 독자적인 차별화를 시도했다.
            [4]
            사실 이 차별화는 당당하고 주체적인 소녀 콘셉트의 정석에 가까운
            ITZY
            , 콘셉트추얼함과 자체 제작으로 승부를 보는
            (여자)아이들
            , 독자적인 세계관과 음악 스타일로 어필하는
            aespa
            등 현재 걸크러시 콘셉트로 활동하는 걸그룹 대부분에게 해당되는 말이긴 하다. 하지만 IVE는 다른 걸 크러시 콘셉트 걸그룹들이 주로 랩, 힙합에 가까운 구성을 갖춘 음악을 내세우는 것과는 달리
            [5]
            타 그룹보다 보컬과 멜로디의 비중이 더 높은 음악을 내세움으로써 좀 더 대중성 있고 트렌디한 음악을 한다
            는 평가를 받는다. 이에 더해, IVE의 노래들은
            2000년대
            후반 ~
            2010년대
            초반에 전성기를 누렸던 2세대 걸그룹들의 노래와도 비슷한 느낌을 많이 주기 때문에 이로 인해 호감을 느끼는 팬들도 상당히 많다.
            연타석 홈런을 친 데뷔 싱글 1집 타이틀곡 <
            ELEVEN
            >과 싱글 2집 타이틀곡 <
            LOVE DIVE
            > 모두 표면적으로 보기에는 흔하게 볼 수 있는 사랑에 빠진 소녀의 심리를 노래한 곡으로 볼 수 있다. 하지만 사실 가사를 자세히 분석해 보면 상대방이 아닌
            자기 자신한테 사랑에 빠진 화자
            의 심리를 노래한다는 상당히 독특한 내용인데, 걸 크러시를 표방한 걸그룹이 이러한 주제를 메인 콘셉트로 잡고 나오는 케이스가 그동안 거의 없었기에 대중들에게 차별화된 콘셉트로 눈도장을 찍을 수 있게 되었다.
            결과적으로 이러한 콘셉트 구축은 매우 성공적인 성과를 냈다. 흔히
            M세대
            후반과
            Z세대
            로 대표되는(1990년대 초중반~2001년대 극초반생) 현 10대 중반~30대 초반의 가장 큰 관심사이자 워너비 소재인
            주체적인 자아상
            , 특히 10대들이라면 누구나 동경할 법한
            자신을 사랑하고 스스로에게 당당한 이미지
            가 제대로 먹혔다. 게다가
            IZ*ONE
            활동을 거치며 쌓아온
            WIZ*ONE
            을 비롯한
            안유진
            과
            장원영
            의 팬덤 화력, 새로운 비주얼 멤버들을 통해 유입된 신규 팬덤의 화력이 더해져 IVE가 빠르게, 그리고 안정적으로 자리를 잡게 해 준 원동력이 되었다.
            물론 IVE도 다른 아이돌들 처럼 일반적인 사랑을 주제로 한 노래가 없는 건 아니다. 타이틀곡으로 대표적으로는
            Off The Record
            ,
            Accendio
            가 있으며, 수록곡들 또한 자기애를 주제로 한 노래와 사랑을 주제로 한 노래가 섞여 있다.
            ```
            
        - **해결:** 나무위키 전용 크롤링 메서드를 만들어보기
            - 나무위키는 크롤링 방지 목적으로 클래스명을 모호하게 짓습니다:
                
                ![image.png](image%201.png)
                
            - 이 무작위 클래스들은 다행히 길이가 `str[8]` (또는 `str[8]-띄어쓰기-str[8]`)이라는 공통점이 있습니다. 따라서 해당 길이의 클래스를 전부 찾아서, 하이퍼링크와 주석만 적절히 처리하는 방법으로 크롤링을 진행해보면 될 것 같습니다.
        
    - ⚠️ trafilatura/readability 라이브러리 이후의 fallback logic이 없음
        
        현재 크롤링 함수는 아래와 같이 되어 있습니다.
        
        ```python
        def crawl(url):
            """
            실제 크롤링 함수 - 여기에 로직을 추가할 수 있음
            """
            # check if the URL matches any known site patterns
            result = dispatch_known_site(url)
            if result:
                return result
        
            # fallback logic (logic2에 해당되는 경우)
            return fallback_extraction(url)
        
        def fallback_extraction(url):
            """
            readability 라이브러리를 사용하여 main content를 추출하는 함수 (fallback logic)
            """
            response = requests.get(url)
            doc = Document(response.text)
            main_content_html = doc.summary()
            soup = BeautifulSoup(main_content_html, 'html.parser')
            filtered_text = soup.get_text(separator='\n', strip=True)
            # print(filtered_text)
            return filtered_text
        ```
        
        - fallback logic인 logic 2를 실패할 경우 오류가 반환되는 문제가 있습니다. 따라서 logic 2에서 오류를 반환할 때의 logic 3를 대비할 필요가 있어 보입니다.
        - input query : “서울대학교 신효필 교수님에 대해 알려줘. 연구분야를 중심으로 알려줘.”
            
            **[오류 메시지]**
            
            ```bash
             File "/home/hyeznee/.local/lib/python3.12/site-packages/lxml/html/__init__.py", line 738, in document_fromstring
                raise etree.ParserError(
            readability.readability.Unparseable: Document is empty
            ```
            
        - 제가 crawl(url) 함수를 수정하는 방향으로 우선 한번 보겠습니다!
        - 지금 보니 진행을 막을 정도로 fatal한 issue는 아닌 것 같네요..
    - ⚠️ 직접 찾아보라는 답변을 하는 경우가 존재함.
        - 문제: search 결과에서 유의미한 내용이 없다고 판단하는 것 같음.
        - 해결 방안 : final_output.py 프롬프트 수정 (진행중)
    - ⚠️ AI 학습 및 활용을 금지하는 웹사이트의 경우
        - 연합뉴스 등의 웹사이트에서 AI 학습 및 활용을 금지한다는 내용을 함께 담고 있음.
        - 해결방안 : crawler.py에서 예외 처리
