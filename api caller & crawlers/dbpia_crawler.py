import requests
from bs4 import BeautifulSoup


def dbpia_handle(url):
    """
    DBpia에서 논문 초록을 뽑아오는 함수. 
    input : url
    output : abstract
    """
    response = requests.get(url)

    # 접속 실패 시
    if response.status_code != 200:
        print(response.status_code)
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 초록만 불러오기
    abstract = soup.find("div", class_ = "abstractTxt").text.strip()
    return abstract

if __name__ == "__main__":
    url = "https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11471821" # sample url
    print(dbpia_handle(url)) # sample query