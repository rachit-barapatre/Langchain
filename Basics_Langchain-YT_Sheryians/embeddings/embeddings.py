from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", dimension=64)

text = [
    "Hello, your name is ChatGPT,",
    "a large language model trained by OpenAI."
]
# Text ko numbers (vector) mein convert kiya
# vector = embeddings.embed_query("hello, world!")
vector = embeddings.embed_documents(text)

# Dikhane ke liye print() function ka use karein!
print("Yeh rahe vector ke pehle 5 numbers:")
print(vector[:5])
# print(len(vector))