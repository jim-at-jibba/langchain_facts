from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str):
        # calculate the embeddings of the query
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them to
        # max_margin_relevance_search_by_vector
        return self.chroma.max_margin_relevance_search_by_vector(embeddings=emb, lambda_mult=0.8)
        return []


    async def aget_relevant_documents(self, query: str):
        return []
