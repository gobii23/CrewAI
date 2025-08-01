import os
import chromadb
import google.generativeai as genai
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tools.csv_index_tool import CsvIndexTool
from pydantic import PrivateAttr

class CsvRAGTool(BaseTool):
    name: str = "CsvRAGTool"
    description: str = (
        "Performs Retrieval-Augmented Generation (RAG) over a structured CSV dataset "
        "to return accurate and grounded answers. It uses embeddings to retrieve the most "
        "relevant rows and uses LLM to generate contextual answers."
    )

    _file_path: str = PrivateAttr()
    _collection_name: str = PrivateAttr()
    _model: any = PrivateAttr()

    def __init__(self, index_tool: CsvIndexTool, **kwargs):
        super().__init__(**kwargs)
        self._file_path = index_tool.file_path
        self._collection_name = index_tool.collection_name
        self._model = genai.GenerativeModel("gemini-2.5-flash")

    def _run(self, query: str) -> str:
        if not os.path.exists(self._file_path):
            return f"File not found: {self._file_path}"

        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        query_embedding = embedding_model.embed_query(query)

        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            collection = client.get_collection(self._collection_name)
        except Exception:
            return f"Collection '{self._collection_name}' not found. Please index it first."

        results = collection.query(query_embeddings=[query_embedding], n_results=10)
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
