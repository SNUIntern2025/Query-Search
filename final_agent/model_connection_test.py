from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI(
        base_url="http://localhost:8000/v1",  # 로컬 vLLM 서버 주소
        api_key="token-snuintern2025"
    )

    completion = client.chat.completions.create(
        model= "snunlp/bigdata_gemma2_9b_dora", #"snunlp/bigdata_exaone3_7.8b_fft", # "snunlp/bigdata_gemma2_9b_dora", : 모델에 따라 변경해줘야 함
        messages=[{"role": "user", "content": "세종대왕에 대해 알려줘."}]
    )

    print(completion.choices[0].message)
