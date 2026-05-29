import torch
from transformers import pipeline

gpu_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU found"

# print(f"GPU Available: {gpu_available}")
# print(f"Device Name: {device_name}")

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype=torch.float16,
    device=0
)

prompt = """
<|user|>
What is AI?

Answer briefly in structured bullet points.
Do not ask follow-up questions.
<|assistant|>
"""

response = pipe(
    prompt,
    max_new_tokens=100,
    temperature=0.6,
    do_sample=False,
    return_full_text=False
);

print(response[0]["generated_text"])
