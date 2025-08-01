import os
import pandas as pd
import chromadb
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import PrivateAttr

class CsvIndexTool(BaseTool):
    name: str = "CsvIndexTool"
    description: str = (
        "Indexes a structured CSV dataset into a ChromaDB vector store using embeddings. "
        "Useful for semantic search on tabular datasets."
    )

    _file_path: str = PrivateAttr()
    _collection_name: str = PrivateAttr()

    def __init__(self, file_path: str, collection_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self._file_path = file_path
        self._collection_name = collection_name or self._derive_name()

    def _derive_name(self):
        return os.path.splitext(os.path.basename(self._file_path))[0].lower().replace(" ", "_")

    @property
    def file_path(self):
        return self._file_path

    @property
    def collection_name(self):
        return self._collection_name

    def _run(self, _: str = None) -> str:
        if not os.path.exists(self._file_path):
            return f"File not found: {self._file_path}"

        df = pd.read_csv(self._file_path, dtype=str).fillna("")
        texts = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
        metadata = df.to_dict(orient="records")

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(self._collection_name)

        if collection.count() > 0:
            return f"Collection '{self._collection_name}' already exists. Skipping indexing."

        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = embedding_model.embed_documents(texts)

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata,
            ids=[f"row{i}" for i in range(len(texts))]
        )

        return f"Indexed {len(texts)} documents into collection '{self._collection_name}'."

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")
