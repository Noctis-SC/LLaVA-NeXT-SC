from openai import OpenAI
from sglang.utils import wait_for_server, print_highlight, terminate_process

client = OpenAI(base_url=f"http://127.0.0.1:35064/v1", api_key="None")

response = client.chat.completions.create(
    model="lmms-lab/LLaVA-Video-7B-Qwen2",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")