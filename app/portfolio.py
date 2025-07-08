import pandas as pd
import chromadb
import uuid
import os


class Portfolio:
    def __init__(self):
        # Load the CSV file
        base_path = os.path.dirname(os.path.abspath(__file__))  # path to /app
        file_path = os.path.join(base_path, "resource", "my_portfolio.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Portfolio CSV not found at: {file_path}")

        self.data = pd.read_csv(file_path)

        # ✅ Initialize ChromaDB in-memory client and collection
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        # Only add documents if the collection is empty
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=[row["Techstack"]],
                    metadatas={"links": row["Links"]},
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        return self.collection.query(
            query_texts=skills,
            n_results=2
        ).get('metadatas', [])
