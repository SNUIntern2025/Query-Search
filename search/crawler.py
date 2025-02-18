# ---------- import libraries ----------

# http request를 위한 라이브러리
import requests
# 정규식을 사용하기 위한 라이브러리
import re
# arvix element tree를 사용하기 위한 라이브러리
import xml.etree.ElementTree as ET
# 웹페이지 파싱을 위한 라이브러리
from bs4 import BeautifulSoup
# fallback logic을 위한 라이브러리
from readability import Document
# 시간 측정용 라이브러리
import time
<<<<<<< HEAD
from urllib.parse import urlparse
=======
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff

# ---------------------------------------


def get_arxiv_abstract(arxiv_id):
    """
    arxvix.org에서 arxiv_id에 해당하는 논문의 초록을 가져오는 함수
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
        abstract = root.find(".//arxiv:summary", ns).text
        return abstract.strip()
    else:
        return fallback_extraction(url)

def handle_arxiv(url):
    """
    arxiv.org 사이트의 URL을 처리하는 함수. 
    주헌 : 별도의 핸들러가 필요한 이유는 arvix_id를 추출해서 export.arxiv.org API를 사용하여 논문의 초록을 가져오도록 하기 위함입니다.
    """
    arxiv_id = url.rstrip('/').split('/')[-1]
    return get_arxiv_abstract(arxiv_id)

def handle_kor_wikipedia(url):
    """
    한국어 위키백과 사이트의 main content를 추출하는 함수
    """
    response = requests.get(url)

    # error handling
    if response.status_code != 200:
        print(f"Error fetching page: {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # find the main content that holds the article text
    main_content = soup.find('div', class_='mw-parser-output')
    if not main_content:
        # if main content container is not found, return to the fallback_extraction
        print("Main content container not found.")
        return fallback_extraction(url)

    for tag in main_content.find_all(['script', 'style', 'table', 'ul', 'div']):
        tag.decompose()

    # extranct only paragraph <p> tags
    paragraphs = main_content.find_all('p')
    # extract plain text from paragraphs
    plain_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return plain_text

def handle_dbpia(url):
    """
    DBpia에서 논문 초록을 뽑아오는 함수. 
    input : url
<<<<<<< HEAD
    return : abstract
=======
    output : abstract
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff
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

def handle_kyobo(url):
    """
    scholar_kyobo에서 논문 초록을 뽑아오는 함수
    input: url
<<<<<<< HEAD
    return: abstract
=======
    output: abstract
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff
    """
    response = requests.get(url)

    # 접속 실패 시
    if response.status_code != 200:
        print(response.status_code)
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    abstract = soup.find("p", class_ = "cont_txt").text
    
    return abstract
<<<<<<< HEAD

def handle_SOF(url):
    """
    StackOverflow에서 main content를 뽑아오는 함수
    """
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching page: {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    main_content = soup.find_all('div', class_ = 's-prose js-post-body')
    extracted_data = []

    if main_content:
        for item in main_content:
            item = item.find_all(["p", "pre"], recursive=False)
            for tag in item: 
                if tag.name == "p":
                    paragraph_text = []
                    for elem in tag.descendants:
                        if elem.name == "a" and elem.get("href"):  
                            paragraph_text.append(f"{elem.get_text(strip=True)} ({elem['href']})")  # Append link with text
                        elif isinstance(elem, str):  
                            paragraph_text.append(elem.strip())  # Append normal text
                    
                    extracted_data.append(" ".join(paragraph_text))


                elif tag.name == "pre":
                    code_tag = tag.find("code")
                    if code_tag:
                        extracted_data.append(code_tag.get_text(strip=True)) 

    else:
        print("Main content container not found.")
        return fallback_extraction(url)
    
    extracted_data = "\n".join(extracted_data)
    return extracted_data

def handle_velog(url):
    """
    velog에서 main content를 뽑아오는 함수
    """
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching page: {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    main_content = soup.body.find('script')
    if main_content:
        main_content = main_content.get_text(strip=True)
        json_str = main_content.split("window.__APOLLO_STATE__=")[-1].strip()  # Extract JSON part
        match = re.search(r'"body"\s*:\s*"((?:\\.|[^"\\])*)"', json_str)
        if match:
            main_content = match.group(1)
            main_content = main_content.replace(r'\n', '\n').replace(r'\t', '\t')
            return main_content

    else:
        print("Main content container not found.")
        return fallback_extraction(url)


def handle_tistory(url):
    """
    tistory에서 main content를 뽑아오는 함수
    """
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching page: {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    main_content = soup.find('div', class_='tt_article_useless_p_margin contents_style')
    if main_content:
        for tag in main_content.find_all(["br"], recursive=False):
            tag.decompose()

    else:
        print("Main content container not found.")
        return fallback_extraction(url)

    for tag in main_content.find_all("pre"): 
        code_tag = tag.find("code")
        if code_tag:
            tag.replace_with(code_tag.get_text(strip=True))
    main_content = main_content.get_text(strip=True)
    
    return main_content  #text 임

def handle_daum_news(url):
    """
    다음 뉴스에서 main content를 뽑아오는 함수
    """
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching page: {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    main_content = soup.find('div', class_='article_view')
    if main_content:
        for tag in main_content.find_all(["figure"]):
            tag.decompose()
        main_content = main_content.get_text(strip=True)

    else:
        print("Main content container not found.")
        return fallback_extraction(url)
    
    return main_content  #text 임

def handle_naver_news(url):
    """
    네이버 뉴스에서 main content를 뽑아오는 함수
    """
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching page: {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # find the main content that holds the article text
    main_content = soup.find('article', class_='go_trans _article_content', id = 'dic_area')
    if main_content:
        for tag in main_content.find_all(["span", "img", "br", "beta"]):
            tag.decompose()
        main_content = main_content.get_text(strip=True)
        

    else:
        print("Main content container not found.")
        return fallback_extraction(url)
    
    return main_content  #text 임

def handle_naver_blog(url):
    """
    네이버 블로그 포스트의 main content를 추출하는 함수.
    포스트의 라이선스 유무를 확인하고, 이용이 불가할 경우 None 반환.
    """
    # Extract blog ID and log number from the URL
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    
    if len(path_parts) < 2:
        print(f"Error fetching page {url}: Invalid Naver blog URL format")
        return None
    
    blog_id = path_parts[0]
    log_no = path_parts[1]
    
    post_view_url = f'https://blog.naver.com/PostView.naver?blogId={blog_id}&logNo={log_no}'
    
    # Set headers to mimic a browser request - prevents redirection to mobile site
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Referer': url
    }

    response = requests.get(post_view_url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Check for existence of a license!
    license_element = soup.select_one('#post_footer_contents > div.wrap_ico_ccl')
    if license_element is not None:
        print(f"Warning: {url}: License invalid for DAG agent use")
        return None
    
    selectors = [
        {'class_': 'se-main-container'},
        {'id': 'postViewArea'},
        {'class_': 'post-view'},
        {'role': 'textbox'}
    ]
    
    main_content = None
    for selector in selectors:
        main_content = soup.find('div', **selector)
        if main_content:
            break
    
    if not main_content:
        print(f"Error fetching page {url}: Content changed")
        return None
    
    text = main_content.get_text(separator='\n', strip=True)
    return text


=======
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff
    
# --- 현재까지 main content의 위치가 파악된 사이트 모음 ---

# 주헌 : 확장성을 고려해서 사이트 패턴과 핸들러를 딕셔너리로 관리하도록 만들어봤습니다.
KNOWN_SITE_HANDLERS = {
    r"arxiv\.org": handle_arxiv,
    r"ko\.wikipedia\.org": handle_kor_wikipedia,
    r"dbpia\.co\.kr/Journal/articleDetail\?": handle_dbpia,
<<<<<<< HEAD
    r"scholar\.kyobobook\.co\.kr/article/detail": handle_kyobo,
    r"n\.news\.naver\.com/article" : handle_naver_news,
    r"v\.daum\.net/v" : handle_daum_news, 
    r"\.tistory\.com": handle_tistory, 
    r"velog\.io/" : handle_velog, 
    r"stackoverflow\.com/": handle_SOF,
    r"blog\.naver\.com": handle_naver_blog
=======
    r"scholar\.kyobobook\.co\.kr/article/detail": handle_kyobo
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff
}

def dispatch_known_site(url):
    """
    URL이 알려진 사이트 패턴과 일치하는지 확인하고 해당되는 웹사이트의 핸들러를 호출하는 함수
    """
    for pattern, handler in KNOWN_SITE_HANDLERS.items():
        if re.search(pattern, url):
            return handler(url)
    return None

def fallback_extraction(url):
    """
    readability 라이브러리를 사용하여 main content를 추출하는 함수 (fallback logic)
    """
    response = requests.get(url)
    doc = Document(response.text)
    main_content_html = doc.summary()
    soup = BeautifulSoup(main_content_html, 'html.parser')
    filtered_text = soup.get_text(separator='\n', strip=True)
<<<<<<< HEAD
    # print(filtered_text)
=======
    print(filtered_text)
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff
    return filtered_text


def crawl(url):
    """
    실제 크롤링 함수 - 여기에 로직을 추가할 수 있음
    주헌 : google search API를 통해 받아온 url을 인자로 사용하면 될 것 같습니다
<<<<<<< HEAD
    args:
        url: 크롤링할 웹사이트의 URL
        return: str, 크롤링한 main content
=======
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff
    """
    # check if the URL matches any known site patterns
    result = dispatch_known_site(url)
    if result:
        return result

    # fallback logic (logic2에 해당되는 경우)
    return fallback_extraction(url)

<<<<<<< HEAD

=======
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff
# -------------- 테스트 코드 -------------------

if __name__ == "__main__":

    # 주헌 : 여기에 테스트 URL을 추가하면 됩니다
    test_urls = [
        "https://arxiv.org/abs/2401.12345", #arxiv 논문 초록
        "https://ko.wikipedia.org/wiki/데이터_매트릭스", #한국어 위키백과
        "https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11471821", # dbpia
        "https://scholar.kyobobook.co.kr/article/detail/4010038753085", # kyobo scholar
<<<<<<< HEAD
        "https://www.yna.co.kr/view/AKR20250204076400009?section=international/all&site=topnews01", #연합뉴스 - readability fallback logic 사용해야함
        "https://stackoverflow.com/questions/79411372/getting-a-2nd-iasyncenumerator-from-the-same-iasyncenumerable-based-on"
        "https://velog.io/@boseung/velog%EA%B0%9C%EB%B0%9C%EB%B8%94%EB%A1%9C%EA%B7%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-%EA%B3%BC%EC%A0%95-%EC%82%BD%EC%A7%88%EA%B8%B0%EB%A1%9D"
        "https://augustfamily.tistory.com/108"
        "https://v.daum.net/v/20250204175018118"
        "https://n.news.naver.com/article/003/0013046572?cds=news_media_pc"
=======
        "https://www.yna.co.kr/view/AKR20250204076400009?section=international/all&site=topnews01"#연합뉴스 - readability fallback logic 사용해야함
>>>>>>> d6fe090190900360adcc8da68c2992545b517fff
    ]

    # 런타임 시간 측정
    start_time = time.time()
    for url in test_urls:
        print(f"\nExtracting content from: {url}")
        content = crawl(url)
        print(content)
        print("-" * 80)

    total_elapsed = time.time() - start_time
    print(f"\nTime taken to crawl {len(test_urls)} websites: {total_elapsed:.2f} seconds")

# --------------------------------------------