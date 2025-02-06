PARALLEL = """
주어진 여러 개의 쿼리에 대답하기 위해, 각각에 대한 추가 정보 검색이 필요한지 판단해야 한다.
정보 검색은 '인터넷 검색'을 통해 수행할 수 있다.

검색 실행 여부에 대한 판단 기준은 다음과 같다:
1. 질문 유형: 질문이 사고를 요구하는가, 정보를 요구하는가?
2. 맥락: 질문에 포함된 정보만으로 답변을 구성할 수 있는가?

인터넷 검색 간 판단 기준은 다음과 같다:
1. 시의성: 주제가 최신 정보와 관련이 있는가? 최신 정보가 답변의 질을 향상시키는가?
2. 정밀성: 답변을 구성하기 위해 구체적이고 정확한 정보가 필요한가?
3. 전문성: 질문의 주제가 전문 지식이나 첨단 연구와 관련이 있는가?
4. 신뢰성: 외부 자료를 통한 교차검증이 답변의 질을 향상시키는가?

답변은 아래 예시와 같은 형식으로 해야 하며, 다른 내용을 포함해서는 안 된다.
routing은 인터넷 검색의 필요 여부를 나타낸다. (필요한 경우 "web", 아닐 경우 "none")
{{"subquery": "대전 주요 관광지 추천", "routing": "web", "reasoning": "대전의 주요 관광지는 시의성이 크고 신뢰성이 중요하므로, 외부 정보를 필요로 합니다."}}
{{"subquery": "지난 주 있었 던 지진에 대해 알려줘", "routing": "web", "reasoning": "지난 주 있었던 지진은 시의성이 크므로, 외부 정보를 필요로 합니다."}},
{{"subquery": "2x + 4 = 0에서 x는?", "routing": "none", "reasoning": "일차방정식의 풀이는 외부 정보를 필요로 하지 않습니다."}}
"""

PARALLEL_ZERO_SHOT= """
주어진 여러 개의 쿼리에 대답하기 위해, 각각에 대한 추가 정보 검색이 필요한지 판단해야 한다.
정보 검색은 '인터넷 검색'을 통해 수행할 수 있다.

검색 실행 여부에 대한 판단 기준은 다음과 같다:
1. 질문 유형: 질문이 사고를 요구하는가, 정보를 요구하는가?
2. 맥락: 질문에 포함된 정보만으로 답변을 구성할 수 있는가?

인터넷 검색 간 판단 기준은 다음과 같다:
1. 시의성: 주제가 최신 정보와 관련이 있는가? 최신 정보가 답변의 질을 향상시키는가?
2. 정밀성: 답변을 구성하기 위해 구체적이고 정확한 정보가 필요한가?
3. 전문성: 질문의 주제가 전문 지식이나 첨단 연구와 관련이 있는가?
4. 신뢰성: 외부 자료를 통한 교차검증이 답변의 질을 향상시키는가?

답변은 아래 예시와 같은 형식으로 해야 하며, 다른 내용을 포함해서는 안 된다.
routing은 인터넷 검색의 필요 여부를 나타낸다.
{{"subquery": "쿼리 1", "is_search": "True", "routing": "True", "reasoning": "쿼리 1에 대한 판단 근거"}}
{{"subquery": "쿼리 2", "is_search": "False", "routing": "False", "reasoning": "쿼리 2에 대한 판단 근거"}}
"""