from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",max_tokens=200,timeout=None,max_retries=2)

response = model.invoke("Write a short poem about the beauty of nature.")
print(response.content)

