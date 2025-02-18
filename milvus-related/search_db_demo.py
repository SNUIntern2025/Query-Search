# demo for searching MilvusDB
# 실제 DB 접근 허락을 아직 받지 못하여, 임의로 collection을 만들어서 demo를 보여드립니다!

# pip install --upgrade pymilvus
# pip install "pymilvus[model]"

from pymilvus import MilvusClient
from pymilvus import model

# setup : Milvus Lite 이용

client = MilvusClient("./milvus_demo.db")

if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
    
client.create_collection(
collection_name="demo_collection",
dimension=768,) # The vectors we will use in this demo has 768 dimensions

# For embedding, used KR-SBERT-Medium-klueNLItriplet_PARpair-klueSTS
sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='snunlp/KR-SBERT-Medium-klueNLItriplet_PARpair-klueSTS', # Specify the model name
    device='cpu' # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

docs = [
    "삼성 갤럭시 S24는 삼성전자가 2024년 1월 17일 공개한 스마트폰이다.",
    "삼성전자의 스마트폰 주력 상품에 해당하며, 삼성 갤럭시 S23(2023. 2. 출시)의 뒤를 잇는 삼성 갤럭시 S 시리즈 15 번째 스마트폰이다.",
    "미국 캘리포니아주 산호세(새너제이)의 SAP 센터에서 열린 '갤럭시 언팩 2024' 행사에서 공개되었다. 전 세계에서 처음으로 생성형 인공지능(generative AI)을 탑재한 것으로 알려졌다.",
    "안드로이드(Android) 운영체제를 탑재했으며, 세로 147mm, 가로 70.6mm과 7.6mm의 두께로 이루어져있다.",
    "갤럭시 S24 시리즈에는 3가지 기기가 포함되며, 이들 기기는 이전 갤럭시 S23 시리즈와 동일한 라인업과 화면 크기를 공유한다.", 
    "보급형 갤럭시 S24는 평면 6.2인치(155mm) 디스플레이를 갖추고 있다.",
    "갤럭시 S24+는 더 큰 6.7인치(168mm) 폼 팩터에 유사한 하드웨어를 갖추고 있다.",
    "라인업의 최상위에 있는 갤럭시 S24 울트라는 평면 6.8인치(173mm) 디스플레이를 갖추고 있다.",
    "S24 및 S24+는 미국, 캐나다, 중국, 마카오, 홍콩, 대만에서는 스냅드래곤 8 Gen 3를, 한국과 일본을 포함한 나머지 국가에서는 삼성 엑시노스 2400을 탑재하고 있으며, S24 울트라는 모든 시장에서 스냅드래곤 8 Gen 3를 탑재하고 있다.",
    "그리고 S24 FE는 삼성 엑시노스 2400의 언더 클럭 버전인 삼성 엑시노스 2400e를 탑재하고 있다.",
    "갤럭시 S24 및 S24+는 알루미늄 버전을 특징으로 하며 앰버 옐로우와 코발트 바이올렛의 네 가지 표준 색상으로 제공되며, 세 가지 추가 색상은 삼성 웹사이트인 제이드 그린(Jade Green)을 통해서만 제공된다.",
    "갤럭시 S24 울트라는 전 세계적으로 갤럭시 칩에 퀄컴의 스냅드래곤 8 Gen 3를 사용하는 반면, S24 및 S24+는 미국, 캐나다, 중국, 홍콩, 대만, 마카오에서만 사용하고 본진인 한국 시장과 일본을 포함한 대부분 지역에서는 삼성 엑시노스 2400을 사용한다.",
    "갤럭시 S24 시리즈는 HDR10+를 지원하는 '다이내믹 LTPO AMOLED 2X' 디스플레이, 2600니트의 최대 밝기, '다이내믹 톤 매핑' 기술을 탑재했다. ",
    "갤럭시 S24 울트라는 디스플레이에 고릴라 글래스 아머 유리를 사용한다.",
    "아이폰 15 프로(iPhone 15 Pro)와 아이폰 15 프로 맥스(iPhone 15 Pro Max)는 애플이 설계하고 개발하고 판매하는 스마트폰이다.",
    "아이폰 14 프로의 뒤를 잇는 17세대 아이폰이다.", 
    "이 기기는 2023년 9월 12일에 캘리포니아 쿠퍼티노에 있는 애플 파크에서 열린 애플 이벤트에서 아이폰 15 및 아이폰 15 플러스, Apple Watch Series 9 및 Apple Watch Ultra 2와 함께 발표되었다.",
    "예약 주문은 9월 15일에 시작되었으며 2023년 9월 22일에 출시되었다.",
    "아이폰 15 및 15 플러스와 마찬가지로 15 프로 및 프로 맥스도 유럽연합의 무선 장비 지침(Radio Equipment Directive)에서 스마트폰에서 USB-C 커넥터의 사용을 의무화하는 것을 준수하기 위해 독점적으로 사용한 라이트닝 커넥터를 USB-C로 교체했다.",
    "아이폰 15 프로 및 프로 맥스는 AV1 비디오 하드웨어 디코딩을 지원한다.",
    "아이폰 15 프로 및 프로 맥스는 A17 Pro 단일 칩 시스템(SoC)이 탑재되어 있다.",
    "아이폰 15 프로와 아이폰 15 프로 맥스의 테두리는 티타늄으로 만들어졌다. ",
    "색상은 내추럴 티타늄, 블루 티타늄, 화이트 티타늄, 블랙 티타늄으로 출시된다.",
    "아이폰 15 프로 및 아이폰 15 프로 맥스는 USB 3.0 전송 속도(최대 10Gbps / 1.25Gbps)를 지원하는 USB-C를 사용한다.[13] 이는 USB 2.0 전송 속도(최대 480Mbps / 60MBps)를 지원하는 아이폰 14 프로 및 아이폰 15/15 플러스 기본 모델보다 향상된 기능이다.",
    "모든 아이폰 15 모델은 최대 4K 해상도의 HDR을 갖춘 USB-C 비디오 출력을 통한 DisplayPort 대체 모드를 지원한다.",
]

docs_embeddings = sentence_transformer_ef.encode_documents(docs)

# Print embeddings
print("Docs_Embeddings:", docs_embeddings)
# Print dimension and shape of embeddings
print("Docs_Dim:", sentence_transformer_ef.dim, docs_embeddings[0].shape)
# Docs_Dim: 768 (768,)

# to demo metadata filtering later.
data = [
    {"id": i, "vector": docs_embeddings[i], "text": docs[i], "subject": "demo_data"}
    for i in range(len(docs))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
# Data has 25 entities, each with fields:  dict_keys(['id', 'vector', 'text', 'subject'])
print("Vector dim:", len(data[0]["vector"]))
# Vector dim: 768

# 데이터 삽입
client.insert(collection_name = "demo_collection", 
              data=data)

# Search: demo queries
examples = [
    {
        "subquery": "애플 아이폰 최신모델",
        "routing": True
    },
    {
        "subquery": "아이폰 15 프로 가격",
        "routing": False
    },
    {
        "subquery": "아이폰 15 프로 맥스 스펙",
        "routing": False
    },
    {
        "subquery": "갤럭시 S24 울트라",
        "routing": True
    },
    {
        "subquery": "갤럭시 S24 출시일",
        "routing": False
    }
]

def encode_routed_subqueries(query_list):
    q = [example["subquery"] for example in query_list if not example["routing"]] # routing == False인 경우
    return sentence_transformer_ef.encode_queries(q)

query_embeddings = encode_routed_subqueries(examples) # 임베딩으로 변환된 쿼리 리스트

# Print embeddings
print("Embeddings:", query_embeddings)

# Print dimension and shape of embeddings
print("Dim:", sentence_transformer_ef.dim, query_embeddings[0].shape)
# Dim: 768 (768,)

# 검색
res = client.search(
    collection_name = "demo_collection",
    data=query_embeddings,  # 여러 개의 쿼리 벡터
    limit=3,  # 결과 개수 제한
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit, "\n", docs[hit["id"]])
        
# TopK results: (아이폰 15 프로 가격)
#  아이폰 15 프로와 아이폰 15 프로 맥스의 테두리는 티타늄으로 만들어졌다. 
#  아이폰 15 프로(iPhone 15 Pro)와 아이폰 15 프로 맥스(iPhone 15 Pro Max)는 애플이 설계하고 개발하고 판매하는 스마트폰이다.
#  아이폰 14 프로의 뒤를 잇는 17세대 아이폰이다.

# TopK results: (아이폰 15 프로 맥스 스펙)
#  아이폰 15 프로와 아이폰 15 프로 맥스의 테두리는 티타늄으로 만들어졌다. 
#  아이폰 15 프로 및 프로 맥스는 A17 Pro 단일 칩 시스템(SoC)이 탑재되어 있다.
#  아이폰 15 프로 및 프로 맥스는 AV1 비디오 하드웨어 디코딩을 지원한다.

# TopK results: (갤럭시 S24 출시일)
#  삼성 갤럭시 S24는 삼성전자가 2024년 1월 17일 공개한 스마트폰이다.
#  삼성전자의 스마트폰 주력 상품에 해당하며, 삼성 갤럭시 S23(2023. 2. 출시)의 뒤를 잇는 삼성 갤럭시 S 시리즈 15 번째 스마트폰이다.
#  갤럭시 S24 시리즈에는 3가지 기기가 포함되며, 이들 기기는 이전 갤럭시 S23 시리즈와 동일한 라인업과 화면 크기를 공유한다.