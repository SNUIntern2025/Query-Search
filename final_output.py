from langchain_community.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from my_utils import timeit
from query.query_with_gemma2 import special_tokens

# TODO: 프롬프트 수정
@timeit
def final_output(query, contexts, llm):
    model_name = llm.model
    prompt = f"""{special_tokens[model_name]["system_start"]} 사용자의 질문에 답하세요.
        {special_tokens[model_name]["user_start"]} {{query}} {special_tokens[model_name]["end_token"]}
        질문에 답할 때 아래 정보를 참고해도 됩니다:
        {{context}} {special_tokens[model_name]["end_token"]}
        {special_tokens[model_name]["assistant_start"]} """

    chat_prompt = PromptTemplate.from_template(prompt)
    chain = (
        chat_prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({'query': query, 'context': '\n'.join(contexts)})

    return answer
    


if __name__ == "__main__":
    query = "날아다니는 파스타 괴물 신화에 대해 알려줘"
    contexts = ['파스타 괴물 신화는 이탈리아의 파스타를 신으로 믿는 종교입니다.',
                '2005년 6월오리건 주립대학의 물리학 학위자인 바비 헨더슨은, 캔자스 교육 위원회가 공립학교의생물학수업에서지적 설계와진화를 동등하게 가르쳐야 한다는 주장에 반응하여 정식적인 공청회를 계획하고 있다는 소식을 듣고, 공개적 인 서한을 보냈다. 바비 헨더슨은 그의 개인 웹사이트[2]에서, 공립학교의 생물학 수업에서 지적 설계나 “증명 가능한 막대한 증거에 기반을 둔 논리적 추측”(진화론)과 마찬가지로 날아다니는 스파게티 괴물교 역시 동등하게 가르쳐져야만 한다고 정식으로 요청했다. 그는 이것이  받아들여지지 않는다면 “우리는 법적 대응을 감행할 수 밖에 없다.”고 경고했다. 얼마 지나지 않아 그는 위원회 위원 세 명으로부터 그 주장에 동조하는 반응[3]을 받았다.']

    model_name = "google/gemma-2-2b-it"
    llm = VLLM(model = "google/gemma-2-2b-it",
            trust_remote_code = True,
            max_new_tokens = 512,
            top_k = 10,
            top_p = 0.95,
            temperature = 0.9,
            gpu_memory_utilization = 0.8, # OOM 방지
            max_num_seqs = 8 # batch size 조정
            # tensor_parallel_size = 4 # for distributed inference
    )

    answer = final_output(query, contexts, llm)
    print(answer)
