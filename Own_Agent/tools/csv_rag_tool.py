import os
import pandas as pd
import chromadb
import google.generativeai as genai
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import PrivateAttr

class CsvRAGTool(BaseTool):
    name: str = "CsvRAGTool"
    description: str = (
        "Indexes a CSV into ChromaDB if needed, retrieves relevant rows using embeddings, "
        "and answers a query using LLM based only on retrieved rows."
    )

    _file_path: str = PrivateAttr()
    _collection_name: str = PrivateAttr()
    _model: any = PrivateAttr()

    def __init__(self, file_path: str, collection_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self._file_path = file_path
        self._collection_name = collection_name or self._derive_name()
        self._model = genai.GenerativeModel("gemini-2.5-flash")
        self._maybe_index_csv()

    def _derive_name(self):
        return os.path.splitext(os.path.basename(self._file_path))[0].lower().replace(" ", "_")

    def _maybe_index_csv(self):
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"CSV file not found: {self._file_path}")

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(self._collection_name)

        if collection.count() > 0:
            return  

        df = pd.read_csv(self._file_path, dtype=str).fillna("")
        texts = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
        metadata = df.to_dict(orient="records")

        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = embedding_model.embed_documents(texts)

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata,
            ids=[f"row{i}" for i in range(len(texts))]
        )

    def _run(self, query: str) -> str:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        query_embedding = embedding_model.embed_query(query)

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(self._collection_name)

        results = collection.query(query_embeddings=[query_embedding], n_results=500)
        documents = results.get("documents", [[]])[0]
        if not documents:
            return "No relevant documents found."

        context = "\n\n".join(documents)

        system_instruction = (
            "You are a CSV RAG assistant. Use only the retrieved context below to answer the query. "
            "Do not make up data. If the answer isn't present, say so."
        )

        response = self._model.generate_content([
            {"role": "model", "parts": [system_instruction]},
            {"role": "user", "parts": [f"Question: {query}\n\nContext:\n{context}"]}
        ])

        return response.text

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")
