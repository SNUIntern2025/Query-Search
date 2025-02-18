import requests
from bs4 import BeautifulSoup


def title_scholar(query):
    """
    Google Scholar에서 논문 제목들을 뽑아오는 함수. 쿼리 형태로 입력받아서 'search'하는 것에 가까움.
    input: query
    output: thesis title + abstract
    """
    url = f"https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&btnG=&q={query}"
    response = requests.get(url)

    # 접속 실패 시
    if response.status_code != 200:
        print(response.status_code)
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    titles = soup.find_all("h3", class_ = "gs_rt") # 주요 정보가 있는 태그, 속성, 속성값
    results = []
    
    for title in titles:
        title_link = title.find("a")
        if title_link is not None:
            title = title_link.text
            link = title_link['href']
            if link[-4:] != ".pdf": # pdf 파일은 링크로 들어가더라도 후처리 방식이 다를 듯하여 제외.
                results.append({"title": title, "link": link})
    
    return results

if __name__ == "__main__":
    print(title_scholar("인공지능")) # sample query