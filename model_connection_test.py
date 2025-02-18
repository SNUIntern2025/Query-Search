from openai import OpenAI
import torch.multiprocessing as mp
import socket
    
def check_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex((host, port))
        if result == 0:
            print(f"포트 {port}에서 연결 가능합니다.")
        else:
            print(f"포트 {port}에서 연결 불가.")
            
if __name__ == "__main__":

        check_port('localhost', 8000)  # 원하는 호스트와 포트 번호로 교체

        client = OpenAI(
            base_url="http://localhost:8000/v1",  # 로컬 vLLM 서버 주소
            api_key="token-snuintern2025"
        )

        completion = client.chat.completions.create(
            model= "snunlp/bigdata_gemma2_9b_dora", #"snunlp/bigdata_exaone3_7.8b_fft", # "snunlp/bigdata_gemma2_9b_dora", : 모델에 따라 변경해줘야 함
            messages=[{"role": "user", "content": "세종대왕에 대해 알려줘."}]
        )

        print(completion.choices[0].message)
