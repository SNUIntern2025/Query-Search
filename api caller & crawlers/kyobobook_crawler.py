import requests
from bs4 import BeautifulSoup


def kyobo_handle(url):
    """
    scholar_kyobo에서 논문 초록을 뽑아오는 함수
    input: url
    output: abstract
    """
    response = requests.get(url)

    # 접속 실패 시
    if response.status_code != 200:
        print(response.status_code)
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    abstract = soup.find("p", class_ = "cont_txt").text
    
    return abstract

if __name__ == "__main__":
    url = "https://scholar.kyobobook.co.kr/article/detail/4010038753085" # sample url
    print(kyobo_handle(url)) # sample query