from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


class TextSplitter:

    def __init__(self, chunk_size=2000, chunk_overlap=0):
        # self.nlp = spacy.load("it_core_news_sm")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    ##TODO SEMANTIC SPLIT REQUIERES SPACY INSTALLED!!!

    # def process(self, text):
    #     doc = self.nlp(text)
    #     sents = list(doc.sents)
    #     vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])
    #     return sents, vecs

    # def cluster_text(self, sents, vecs, threshold):
    #     clusters = [[0]]
    #     for i in range(1, len(sents)):
    #         if np.dot(vecs[i], vecs[i-1]) < threshold:
    #             clusters.append([])
    #         clusters[-1].append(i)

    #     return clusters

    # def clean_text(self, text):
    #     # Add your text cleaning process here
    #     return text

    # def semantic_split_text(self, data, threshold=0.3):
    #     '''
    #     Split thext using semantic clustering and spacy see https://getpocket.com/read/3906332851
    #     '''

    #     # Initialize the clusters lengths list and final texts list
    #     clusters_lens = []
    #     final_texts = []

    #     # Process the chunk
    #     sents, vecs = self.process(data)

    #     # Cluster the sentences
    #     clusters = self.cluster_text(sents, vecs, threshold)

    #     for cluster in clusters:
    #         cluster_txt = self.clean_text(' '.join([sents[i].text for i in cluster]))
    #         cluster_len = len(cluster_txt)

    #         # Check if the cluster is too short
    #         if cluster_len < 60:
    #             continue

    #         # Check if the cluster is too long
    #         elif cluster_len > 3000:
    #             threshold = 0.6
    #             sents_div, vecs_div = self.process(cluster_txt)
    #             reclusters = self.cluster_text(sents_div, vecs_div, threshold)

    #             for subcluster in reclusters:
    #                 div_txt = self.clean_text(' '.join([sents_div[i].text for i in subcluster]))
    #                 div_len = len(div_txt)

    #                 if div_len < 60 or div_len > 3000:
    #                     continue

    #                 clusters_lens.append(div_len)
    #                 final_texts.append(div_txt)

    #         else:
    #             clusters_lens.append(cluster_len)
    #             final_texts.append(cluster_txt)

    #     #converting to Langchain documents
    #     ##lo posso fare anche con .create_documents !!
    #     # final_docs=[]
    #     # for doc in final_texts:
    #     #     final_docs.append(Document(page_content=doc, metadata={"source": "local"}))

    #     return final_texts

    def create_langchain_documents(self, texts, metadata):
        final_docs = []
        if type(texts) == str:
            texts = [texts]
        for doc in texts:
            final_docs.append(Document(page_content=doc, metadata=metadata))
        return final_docs

    # fixed split
    def fixed_split(self, data):
        """
        Takes Langchain documents as input
        Returns splitted documents
        """
        docs = self.text_splitter.split_documents(data)
        return docs
