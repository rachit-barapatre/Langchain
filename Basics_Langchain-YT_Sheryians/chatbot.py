from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_tokens=200)

print("---------------------WELCOME TO CHATBOT---------------------")
print("----------------------Type 0 to exit---------------------")
while True:
    prompt = input("YOU: ")
    if prompt == "0":
        break
    response = model.invoke(prompt)

    print("BOT:",response.content)