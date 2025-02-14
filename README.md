# Query-Search

# Query Part (자세한 설명은 [여기](https://www.notion.so/199245846c0a8000b2b4da50f44475db?pvs=21)에)

- 경로: `Query+Search_vllm/query`  안의 모든 파일들

### `query_pipeline.py`

- 서브쿼리 분리 및 쿼리 라우팅을 실행하는 함수
- 아래 코드를 통해 쿼리 파이프라인만 실행해볼 수 있습니다.

```jsx
CUDA_VISIBLE_DEVICES=0 python query_pipeline.py --vllm=true
```

### `query_with_gemma2.py`

- 서브쿼리 분리를 실행하는 함수

### `parallel.py`

- 쿼리 라우팅을 진행하는 함수

# Search Part (자세한 설명은 [채은님 노션](https://www.notion.so/197245846c0a80bcad60d3c10dfd2812?pvs=21)에)

- 경로: `Query+Search_vllm/search`  안의 모든 파일들

### `search_pipeline.py`

- 서브쿼리를 받아 검색을 수행하고, 크롤링을 해 오고, 이를 요약하는 함수
- 아래 코드를 통해 서치 파이프라인만 실행해볼 수 있습니다. (vllm dafault)
    - 스파게티 괴물 신화 관련 내용이 디폴트 값으로 들어갑니다.

```jsx
CUDA_VISIBLE_DEVICES=0 python search_pipeline.py
```

### `serper.py`

- google serper api를 이용하여 검색을 수행하는 함수
- 비동기적으로 동작
- 일단 `K` (검색 개수) 값은 `k=1` 로 지정

### `crawler.py`

- 여러 웹사이트들에 접속하여 내용을 가져오는 함수
- 비동기적으로 동작

### `summarizer.py`

- 가져온 인터넷 정보들을 요약하는 함수
- 한 인터넷 정보가 한 llm의 input으로 들어감 (쿼리 라우팅과 비슷)
    
    → 그러므로 비동기로 실행..해야 했으나 vllm관련 이슈로 vllm에서는 비동기 실행 x
    

# Final output part

### `final_output.py`

- 가져온 모든 인터넷 정보들을 통합하여 사용자의 답변을 내는 함수
- 아래 코드를 실행하여 해당 코드만 실행해볼 수 있습니다.
    - vllm이 디폴트로 들어감

```jsx
CUDA_VISIBLE_DEVICES=0 python final_output.py
```

### `main.py`

- 모든 pipeline이 한 군데 모이는 함수
- 아래의 코드를 통해 실행 가능

```jsx
CUDA_VISIBLE_DEVICES=0 python final_output.py --vllm=true
```

> **인자 설명**
- `vllm` : vLLM을 사용할지 말지 결정하는 인자. “true”를 주면 vLLM으로 wrapping 된 채로 실행됩니다.
> 

# 기타 함수

### `my_utils.py`

- 특정 함수의 소요 시간을 측정하는 데코레이터 `timeit` 존재
- 코드
    
    ```jsx
    from datetime import datetime
    
    # 함수 소요시간 측정 decorator
    def timeit(method):
        def timed(*args, **kw):
            ts = datetime.now()
            result = method(*args, **kw)
            te = datetime.now()
            print(f"{method.__name__} 소요시간: {te-ts}")
            return result
        return timed
    ```
    
- 해당 데코레이터를 함수 위에 붙여주면, 해당 함수가 동작하는 데 걸린 총 시간을 측정하여 출력 해 줍니다.
    - 사용 예시
    
    ```jsx
    @timeit
    def query_pipeline(query, model_name, llm, is_vllm):
    ...
    ```
    
    - 결과
  ```
  query_pipeline 소요 시간: ...
