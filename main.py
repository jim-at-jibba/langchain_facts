from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()

# It will first take a chunk of text the size of the chunk_size,
# then it will split it by the next separator
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0 # copies a part of the previous chunk to the next one
)
loader = TextLoader("facts.txt")

docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

results = db.similarity_search("What interesting fact about the human body?")

for result in results:
    print("\n")
    print(result.page_content)
