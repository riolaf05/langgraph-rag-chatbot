from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Embedder
class EmbeddingFunction:
    def __init__(self, parametro):
        self.parametro = parametro

        if parametro == "openAI":
            # self.embedder = OpenAIEmbeddings(model="text-embedding-3-large")
            self.embedder = OpenAIEmbeddings()

        # elif parametro == 'sentenceTransformers':
        #     self.embedder = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')

        elif parametro == "bgeEmbedding":
            model_name = "BAAI/bge-small-en-v1.5"
            encode_kwargs = {
                "normalize_embeddings": True
            }  # set True to compute cosine similarity
            bge_embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda"},
                encode_kwargs=encode_kwargs,
            )
            self.embedder = bge_embeddings

        elif parametro == "fast-bgeEmbedding":
            self.embedder = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        elif parametro == "hkunlpEmbedding":
            model_name = "hkunlp/instructor-large"
            encode_kwargs = {
                "normalize_embeddings": True
            }  # set True to compute cosine similarity
            bge_embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda"},
                encode_kwargs=encode_kwargs,
            )
            self.embedder = bge_embeddings
        # else:
        #     self.embedder = self.default_method
