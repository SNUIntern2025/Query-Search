import serper as serper
import crawler as crawler
import concurrent.futures
import summarizer as summarizer
import asyncio
import time
import torch
import torch.distributed as dist
import os, re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from search_pipeline import search_pipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline


if __name__ == "__main__":
    query = "날아다니는 파스타 괴물 신화에 대해 알려줘"

    processed_query = [
    {
        "subquery": "날아다니는 파스타 괴물",
        "routing": "web"
    },
    {
        "subquery": "파스타 괴물 신화",
        "routing": "web"
    }
    ]
    model_name = "google/gemma-2-2b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                            torch_dtype=torch.bfloat16, 
                                             device_map="auto", # accelerator 사용용.
                                             use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,  
        max_new_tokens=256,  # 512 > 256
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    # Langchain에 넣어주기 위해서 pipeline으로 감싸기
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    
    
    # Summarization 거친 결과물
    summarized_results = search_pipeline(processed_query, llm)
    
    contexts = []
    for result in summarized_results:
        contexts.append(result['output_text'])

    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=f"""아래 정보에 기반하여, 사용자의 질문에 답하세요.
        {contexts}
        사용자 질문: {query} """
    )           

    chat_prompt = prompt_template.format(context="\n".join(contexts), query=query)

    # runnable로 감싸기
    chat_runnable = RunnableLambda(lambda input: chat_prompt)
    
    chain = (
        chat_runnable
        | llm
        | StrOutputParser()
    )

    start_time = time.time()
    answer = chain.invoke(query)
    end_time = time.time()

    print(answer, f"Execution Time: {end_time-start_time}")