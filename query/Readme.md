# Query 파트

# Sub Query 분해

- EXAONE-3.5-2.4B로 구현
    - 쓸데없이 많은 개수의 결과를 출력하는 경향이 있어 프롬프트로 조정
- 거기에 vLLM을 붙여 속도를 더 빠르게 함
- 실행 시간 2.5~4초 → 0.5~1초로 단축
- 코드 위치: `Query/query_with_gemma2.py`

# Query Routing

- EXAONE-3.5-2.4B로 구현
    - 서브쿼리 분해와 동일한 모델 사용
- 코드 위치: `Query/parallel.py`
- 기존: 여러 서브쿼리들이 각각 하나씩 LLM으로 들어가고, 이를 병렬처리하여 시간을 아끼는 구조

<aside>
⚠️

그러나 여기서 문제 발생!

채은님 노션에 쓰인 내용처럼 vLLM과 비동기 처리가 호환되지 않음.

채은님처럼 두 가지 상황을 비교

→ vLLM 有 + 동기 vs vLLM 無 + 비동기

</aside>

### vLLM 有 + 동기

- 순서대로 작업을 처리하므로, 서브쿼리 개수 (그 중 LLM에 넘겨야겠다고 판단된 서브쿼리 개수) 에 따라 천차만별
- 보통 서브쿼리 3개를 처리할 경우, 약 2.5-3초정도가 걸림

### vLLM 無 + 비동기

- 서브쿼리 3개 기준 9.5-10초 정도가 걸림
- 비동기를 감안하더라도 vLLM보다 느림

<aside>
📝

따라서,

vLLM 有 + 동기를 쓰는 것이 더 바람직함.

병렬 처리 같은 경우는 추후에 지원되지 않을까 추측

</aside>

# 전체 Query Pipeline

- Query가 넘어오면 → 이를 subquery로 분해하고 → 각 subquery를 routing하는 구조
- 이 때 실행 시간은 다음 시간의 합과 같음:
    - sub query 분해 시간 (0.5-1초)
    - rule-based query routing 시간 (0.2초 이하)
    - query routing 시간 (2.5-4초)
    
    **⇒ 전체 3초-5초 소요**
    
- 코드 위치: `Query/query_pipeline.py`
- 실행 코드
    
    ```jsx
     CUDA_VISIVIE_DEVICES=0 python query_pipeline.py
    ```
    
    - 만일, vllm을 끄고 실행해보고 싶다면
    
    ```jsx
      CUDA_VISIVIE_DEVICES=0 python query_pipeline.py --vllm=false
    ```
    
    이렇게 해 주시면 됩니다!
