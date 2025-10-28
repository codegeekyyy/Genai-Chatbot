from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# Initialize model
model = OllamaLLM(model="gemma3:4b", temperature=0.1)

# Initialize parser
parser = StrOutputParser()

def chat():
    chathistory = [
        {"role": "system", "content": "You are a helpful chatbot. Be concise and accurate."}
    ]

    print("Langchain Chatbot. Type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        # Add user message correctly (use dict, not tuple)
        chathistory.append({"role": "user", "content": user_input})

        # Build prompt
        prompt = ChatPromptTemplate.from_messages(chathistory)

        # Create chain
        chain = prompt | model | parser

        # Invoke
        res = chain.invoke({})

        print(f"Bot: {res}\n")

        # Add assistant message to chat history
        chathistory.append({"role": "assistant", "content": res})

        print("-" * 80)

chat()
