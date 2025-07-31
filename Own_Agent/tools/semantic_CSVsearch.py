import pandas as pd
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from crewai.tools import BaseTool

class AutoSemanticCSVTool(BaseTool):
    name: str = "SemanticSearchTool"
    description: str = "Uses semantic search to find insights from a CSV and outputs PlotTool-compatible JSON."

    def __init__(self, csv_path: str):
        super().__init__()
        self._csv_path = csv_path
        self._df = pd.read_csv(self._csv_path, low_memory=False)
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._prepare_embeddings()

    def _prepare_embeddings(self):
        text_columns = self._df.select_dtypes(include=['object', 'string']).columns
        self._df["_combined_text"] = self._df[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
        self._texts = self._df["_combined_text"].tolist()

        self._embeddings = self._model.encode(
            self._texts, convert_to_numpy=True, show_progress_bar=True
        )
        dim = self._embeddings.shape[1]

        self._index = faiss.IndexFlatL2(dim)
        self._index.add(self._embeddings)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_embedding = self._model.encode([query], convert_to_numpy=True)
        distances, indices = self._index.search(query_embedding, top_k)
        return self._df.iloc[indices[0]]

    def _auto_detect_columns(self, query: str):
        text_cols = self._df.select_dtypes(include=['object', 'string']).columns
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns

        query_vector = self._model.encode([query])[0]

        best_label, best_value = None, None
        best_label_score, best_value_score = -1, -1

        for col in text_cols:
            col_vector = self._model.encode([col])[0]
            score = query_vector @ col_vector
            if score > best_label_score:
                best_label_score = score
                best_label = col

        for col in numeric_cols:
            col_vector = self._model.encode([col])[0]
            score = query_vector @ col_vector
            if score > best_value_score:
                best_value_score = score
                best_value = col

        return best_label, best_value

    def to_plot_format(self, query: str, top_k: int = 5) -> dict:
        results = self.search(query, top_k)
        label_col, value_col = self._auto_detect_columns(query)

        labels = results[label_col].astype(str).tolist()
        values = pd.to_numeric(results[value_col], errors="coerce").fillna(0).tolist()

        print(f"Auto-selected columns -> Label: {label_col}, Value: {value_col}")

        return {
            "labels": labels,
            "values": values,
            "title": f"Results for: {query}",
            "xlabel": label_col,
            "ylabel": value_col
        }

    def _run(self, query: str) -> str:
        output = self.to_plot_format(query)
        return json.dumps(output)
