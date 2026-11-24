import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)

# conversation = client.chat.completions.create(
#     model="google/gemma-4-26b-a4b-it",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of Indonesia?"},
#     ],
# )

# print(conversation.choices[0].message.content)

response = client.embeddings.create(
    input="Your text string goes here", model="amazon/titan-embed-text-v1"
)

print(response)
print(len(response.data[0].embedding))
