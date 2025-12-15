from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import os


class MilvusDB:
    """
    Milvus database wrapper that dynamically infers schema fields from metadata.
    """
    
    # Maximum length for VARCHAR fields
    VARCHAR_MAX_LENGTH = 8192

    def __init__(
        self,
        db_path: str = "milvus.db",
        embedding_dim: int = 512,
        host: str = "localhost",
        port: str = "19530",
        auto_extend_schema: bool = True
    ):
        """
        Initialize Milvus database connection.

        Args:
            collection_name: Milvus collection name
            embedding_dim: Dimension of embedding vectors
            host: Milvus host
            port: Milvus port
            auto_extend_schema: If True, auto-add new metadata fields not yet in schema
        """
        self.collection_name = os.path.basename(db_path).replace(".db", "")
        self.embedding_dim = embedding_dim
        self.auto_extend_schema = auto_extend_schema

        # Connect to Milvus
        connections.connect(alias=f"milvus_{self.collection_name}", host="milvus-lite", port=port, uri=db_path)
        # connections.connect(alias="default", host=host, port=port)

        self.collection = None
        if utility.has_collection(self.collection_name, using=f"milvus_{self.collection_name}"):
            self.collection = Collection(self.collection_name, using=f"milvus_{self.collection_name}")
            print(f"🔄 Loaded existing collection: {self.collection_name}")
        else:
            print(f"⚠️ Collection {self.collection_name} does not exist yet. Will create on first insert.")

    # ------------------ Utility methods ------------------ #
    def _infer_field_type(self, value: Any) -> DataType:
        """Infer Milvus data type from Python value."""
        if isinstance(value, bool):
            return DataType.BOOL
        elif isinstance(value, int):
            return DataType.INT64
        elif isinstance(value, float):
            return DataType.FLOAT
        else:
            return DataType.VARCHAR

    def _create_collection_from_metadata(self, metadata: Dict[str, Any]):
        """Create a Milvus collection dynamically based on metadata keys."""
        print(f"🛠️ Creating collection {self.collection_name} dynamically...")

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]

        # Add metadata fields dynamically
        for key, value in metadata.items():
            dtype = self._infer_field_type(value)
            if dtype == DataType.VARCHAR:
                fields.append(FieldSchema(name=key, dtype=dtype, max_length=self.VARCHAR_MAX_LENGTH))
            else:
                fields.append(FieldSchema(name=key, dtype=dtype))

        schema = CollectionSchema(fields, description="Dynamic embedding storage")
        self.collection = Collection(name=self.collection_name, schema=schema, using=f"milvus_{self.collection_name}")

        print(f"✅ Created new collection: {self.collection_name}")
        self._ensure_index()

    def _ensure_index(self):
        """Ensure embedding index exists."""
        if not self.collection.indexes:
            print("⚙️ Creating index for embeddings...")
            self.collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            )
        self.collection.load()

    def _extend_schema_if_needed(self, metadata: Dict[str, Any]):
        """
        Add new fields dynamically to the schema if not present.
        This requires Milvus 2.4+ dynamic schema feature enabled.
        """
        existing_fields = {f.name for f in self.collection.schema.fields}
        new_fields = {k: v for k, v in metadata.items() if k not in existing_fields}

        if new_fields:
            if not self.auto_extend_schema:
                print(f"⚠️ New fields detected but auto_extend_schema=False: {list(new_fields.keys())}")
                return
            print(f"➕ Adding new fields dynamically: {list(new_fields.keys())}")
            for k, v in new_fields.items():
                dtype = self._infer_field_type(v)
                if dtype == DataType.VARCHAR:
                    self.collection.schema.fields.append(FieldSchema(name=k, dtype=dtype, max_length=self.VARCHAR_MAX_LENGTH))
                else:
                    self.collection.schema.fields.append(FieldSchema(name=k, dtype=dtype))
            self.collection.flush()

    # ------------------ Core Methods ------------------ #
    def add_embedding(self, embedding: np.ndarray, id: str, metadata: Dict[str, Any]) -> int:
        """Insert a new embedding + metadata into Milvus."""
        if self.collection is None:
            self._create_collection_from_metadata(metadata)
        elif self.auto_extend_schema:
            self._extend_schema_if_needed(metadata)

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        embedding = (embedding / np.linalg.norm(embedding)).astype(np.float32)

        fields = [f.name for f in self.collection.schema.fields if f.name not in ("pk",)]
        insert_data = []
        for name in fields:
            if name == "embedding":
                insert_data.append([embedding.tolist()])
            elif name == "id":
                insert_data.append([id])
            else:
                val = metadata.get(name, None)
                if isinstance(val, (list, tuple)):
                    val = ",".join(map(str, val))
                # Truncate string values that exceed VARCHAR max length
                if isinstance(val, str):
                    field_schema = next((f for f in self.collection.schema.fields if f.name == name), None)
                    if field_schema and field_schema.dtype == DataType.VARCHAR:
                        max_len = field_schema.max_length or self.VARCHAR_MAX_LENGTH
                        original_len = len(val)
                        if original_len > max_len:
                            val = val[:max_len]
                            print(f"⚠️ Truncated field '{name}' from {original_len} to {max_len} characters")
                insert_data.append([val])

        result = self.collection.insert(insert_data)
        self.collection.flush()
        return result.primary_keys[0]
    

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[Tuple[int, float, dict]]:
        """Search for nearest embeddings, optionally filtered by metadata."""
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        query_embedding = (query_embedding / np.linalg.norm(query_embedding)).astype(np.float32)

        output_fields = [f.name for f in self.collection.schema.fields if f.name not in ("embedding", "pk")]
        results = self.collection.search(data=[query_embedding.tolist()], anns_field="embedding", param={"metric_type": "IP", "params": {"nprobe": 16}}, limit=k, expr=filter_expr, output_fields=output_fields)

        output = []
        for hit in results[0]:
            meta = {f: hit["entity"].get(f) for f in output_fields if f in hit["entity"]}
            output.append((hit.id, float(hit.distance), meta))
        return output

    def query(self, expr: str) -> List[Dict[str, Any]]:
        """Run a metadata-only query."""
        output_fields = [f.name for f in self.collection.schema.fields if f.name not in ("embedding", "pk")]
        results = self.collection.query(expr=expr, output_fields=output_fields)
        return results

    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve by custom ID field."""
        expr = f'id == "{id}"'
        res = self.query(expr)
        return res[0] if res else None

    def delete_by_id(self, id: str) -> bool:
        """Delete by ID."""
        expr = f'id == "{id}"'
        deleted = self.collection.delete(expr)
        self.collection.flush()
        return deleted.delete_count > 0


if __name__ == "__main__":
    db = MilvusDB(collection_name="dynamic_embeddings", embedding_dim=512)

    # Add embedding with custom metadata
    embedding = np.random.rand(512)
    metadata = {
        "duration_start": 3.5,
        "duration_end": 5.2,
        "objects": ["car", "person"],
        "scene": "street",
        "weather": "sunny"
    }
    pk = db.add_embedding(embedding, id="clip_001", metadata=metadata)
    print("Inserted with PK:", pk)

    # Search by embedding and filter
    query = np.random.rand(512)
    results = db.search(query, k=3, filter_expr='scene == "street" and weather == "sunny"')
    print("*****************Results:*****************")
    for r in results:
        print(r)

    # Query only by metadata
    print("*****************Query by metadata:*****************")
    print(db.query('objects like "%person%"'))

    # Get by ID
    print("*****************Get by ID:*****************")
    print(db.get_by_id("clip_001"))
