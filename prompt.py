from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import langchain

# langchain.debug = True

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# Create a new Chroma instance
db = Chroma(
    embedding_function=embeddings,
    persist_directory="emb")

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
)

result = chain.run("What is an interesting fact about the english language?")

print(result)
