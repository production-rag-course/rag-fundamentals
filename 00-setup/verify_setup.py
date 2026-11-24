import os
from importlib.metadata import version

from dotenv import load_dotenv

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")

    print("Setup complete!")


if __name__ == "__main__":
    main()
