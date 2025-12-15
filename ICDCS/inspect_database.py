#!/usr/bin/env python3
"""Script to inspect existing databases"""
import sys
sys.path.append('embeddings')
from SQLiteDB import SQLiteDB
from Milvus import MilvusDB
import json

def inspect_databases(object_faiss_path, event_faiss_path, sqlite_path):
    print("="*60)
    print("DATABASE INSPECTION REPORT")
    print("="*60)
    
    # Inspect SQLite
    print("\n📊 SQLITE DATABASE")
    sqlite_db = SQLiteDB(sqlite_path)
    
    # Get sample objects
    objects = sqlite_db.get_all_tracked_objects()
    print(f"Total tracked objects: {len(objects)}")
    if objects:
        print("\n🔍 Sample Object:")
        print(json.dumps(objects[0], indent=2, default=str))
    
    # Inspect Object Milvus
    print("\n\n🔍 OBJECT MILVUS DATABASE")
    obj_db = MilvusDB(object_faiss_path, 768)  # JinaCLIP dimension
    if obj_db.collection:
        print(f"Collection: {obj_db.collection_name}")
        print(f"Total entries: {obj_db.collection.num_entities}")
        # Sample search
        import numpy as np
        sample = obj_db.search(np.random.rand(768), k=1)
        if sample:
            print("\n🔍 Sample Entry:")
            print(json.dumps(sample[0][2], indent=2, default=str))
    
    # Inspect Event Milvus
    print("\n\n📅 EVENT MILVUS DATABASE")
    evt_db = MilvusDB(event_faiss_path, 768)
    if evt_db.collection:
        print(f"Collection: {evt_db.collection_name}")
        print(f"Total entries: {evt_db.collection.num_entities}")
        sample = evt_db.search(np.random.rand(768), k=1)
        if sample:
            print("\n🔍 Sample Entry:")
            print(json.dumps(sample[0][2], indent=2, default=str))
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python inspect_database.py <object_db> <event_db> <sqlite_db>")
        sys.exit(1)
    
    inspect_databases(sys.argv[1], sys.argv[2], sys.argv[3])