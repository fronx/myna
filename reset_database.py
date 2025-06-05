from qdrant_client import QdrantClient
from qdrant_client.models import Filter

client = QdrantClient(url="http://localhost:6333")

collection_name = "myna_embeddings_new"
client.delete(
    collection_name=collection_name,
    points_selector=Filter(must=[])
)
