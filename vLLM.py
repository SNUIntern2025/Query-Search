from vllm import LLM
import time

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
outputs = llm.generate(prompts)  # Generate texts from the prompts.

start_time = time.time()
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
end_time = time.time()

print(f"Execution Time: {end_time - start_time:.2f}s")

# Prompt: 'Hello, my name is', Generated text: ' Dustin Nelson and I am a senior at Dublin Coffman High School'
# Prompt: 'The capital of France is', Generated text: ' Paris, but it’ located at 1600 Pembina'
# Execution Time: 0.00s