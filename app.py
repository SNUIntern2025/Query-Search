import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from query.query_pipeline import query_pipeline
from search.search_pipeline import search_pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
from langchain_community.llms import VLLM
from final_output import final_output
import argparse
import time
from datetime import datetime
from main import load_model, load_vllm

#전역 변수 설정
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
parser = argparse.ArgumentParser()
parser.add_argument('--vllm', type=str, default='true', help='Using vLLM or not')
args = parser.parse_args()
load_func = load_vllm if args.vllm == 'true' else load_model

try:
    llm = load_func(MODEL_NAME)
except Exception as e:
    print(f"모델 로딩 실패: {e}")
    raise

with gr.Blocks() as demo:
    #제목 설정
    gr.Markdown("# Chat with a DAG Agent and see its thoughts 💭")

    chatbot = gr.Chatbot(
        type = "messages",
        label = "Agent",
        height=600
        ) #챗봇 컴포넌트

    with gr.Row():
        msg = gr.Textbox(
            lines=3,
            placeholder="DaG Agent에게 질문해보세요.\nex)이번 주 토요일에 부산으로 여행을 가려는데 날씨가 괜찮을까? 또 거기서 가볼만한 곳 추천해줘",
            scale=9) # 사용자 입력 칸
        chat_btn = gr.Button("➤", scale=1, min_width=50) #제출 버튼

    clear = gr.ClearButton([msg, chatbot]) # 초기화 버튼

    # 사용자 메시지를 처리하는 함수: 텍스트박스를 비우고, 사용자 메시지를 기록에 추가
    def user(user_message, history: list):
        if not user_message.strip():  # 빈 메시지 체크
            return "", history
        return "", history + [{"role": "user", "content": user_message}] 
    
    #응답 생성 함수
    def respond(history: list):
        if not history:
            yield history
            return
        try:
            user_message = history[-1]['content']

            subqueries, processed_query = query_pipeline(user_message, MODEL_NAME, llm, args.vllm)

            #서브쿼리 결과 표시
            history.append({
            "role": "assistant",
            "content": "\n".join([f"- {query}" for query in subqueries]),
            "metadata": {"title": "🤔 질문 분해 결과"}
            })
            yield history

            #라우팅 결과 표시
            routing_result = "\n".join([f"- {query['subquery']}: {query['routing']}" for query in processed_query])
            history.append({
            "role": "assistant",
            "content": routing_result,
            "metadata": {"title": "🔄 라우팅 결과"}
            })
            yield history

            # 검색 단계 표시
            history.append({
            "role": "assistant",
            "content": "검색 중...",
            "metadata": {"title": "🔍 검색 진행 중"}
            })
            yield history

            # 검색 수행
            search_result = search_pipeline(processed_query, llm, args.vllm)
            answer = final_output(user_message, search_result, llm)

            #스트리밍 효과
            history.append({"role": "assistant", "content": ""})
            for character in answer:
                history[-1]['content'] += character
                time.sleep(0.05)
                yield history

        except Exception as e:
            print(f"에러 발생 {e}")
            history.append({
                "role": "assistant",
                "content": f"처리 중 오류가 발생했습니다.",
                "metadata": {"title": "❌ 오류 발생"}
            })
            yield history
    
    #사용자 메세지-응답 생성 체인 설정
    msg.submit( #Enter키를 눌렀을 때
        user,
        [msg, chatbot],
        [msg, chatbot],
        queue = False
    ).then(
        respond,
        chatbot,
        chatbot
    )

    chat_btn.click( #제출 버튼을 눌렀을 때
        user,
        [msg, chatbot],
        [msg, chatbot],
        queue = False
    ).then(
        respond,
        chatbot,
        chatbot
    )

    #초기화 버튼 성정
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)