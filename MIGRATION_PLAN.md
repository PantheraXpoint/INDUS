# Migration Plan: Replace MilvusDB with AVA Vector Databases

## Overview
Replace the current MilvusDB-based system in ICDCS with AVA's vector database system (BaseVectorStorage). The goal is to use the same algorithm but with AVA's database infrastructure for events and entities (2 views instead of AVA's 3 views).

---

## Current Architecture Analysis

### ICDCS (Current - MilvusDB)
**Location**: `ICDCS/graph_interfaces.py`

1. **KnowledgeGraphInterface**:
   - `self.object_faiss_db = MilvusDB(...)` - Object/entity storage
   - `self.event_faiss_db = MilvusDB(...)` - Event storage
   - Methods:
     - `search_events_by_description(query_text, top_k)` → Returns `List[Node]`
     - `search_objects_by_description(query_text, top_k)` → Returns `List[Node]`
     - `search_events_by_embedding(embedding, top_k)` → Returns `List[Node]`
     - `search_objects_by_embedding(embedding, top_k)` → Returns `List[Node]`
     - `get_objects_in_event(event_id)` → Returns `List[str]` (object IDs)
     - `get_events_containing_object(object_id)` → Returns `List[str]` (event IDs)

2. **Usage in graph_engine.py** (Lines 957-988):
   ```python
   init_events = self.kg.search_events_by_description(keywords_response, top_k=45)
   init_objects = self.kg.search_objects_by_description(rewrite_entity_response, top_k=45)
   ```

### AVA (Target - BaseVectorStorage)
**Location**: `AVA/ava.py`, `AVA/tree_search.py`

1. **Vector Databases**:
   - `self.events_vdb = TextNanoVectorDBStorage(...)` - Event storage
   - `self.entities_vdb = TextNanoVectorDBStorage(...)` - Entity storage (equivalent to objects)
   - `self.features_vdb = ImageNanoVectorDBStorage(...)` - Visual features (NOT USED in ICDCS)

2. **Methods**:
   - `events_vdb.query(query_text, top_k)` → Returns `List[Dict]` with structure:
     ```python
     {
         "id": "event_id",
         "description": "...",
         "duration": [...],
         "__metrics__": similarity_score,
         ...
     }
     ```
   - `entities_vdb.query(query_text, top_k)` → Returns `List[Dict]` with structure:
     ```python
     {
         "id": "entity_id",
         "descriptions": [...],
         "events": [...],  # List of event IDs this entity appears in
         "timestamps": [...],
         "__metrics__": similarity_score,
         ...
     }
     ```
   - `events_vdb.get_data(event_id)` → Returns single dict
   - `entities_vdb.get_data(entity_id)` → Returns single dict

3. **Usage in tree_search.py** (Lines 70-75):
   ```python
   events_result = events_vdb.query(keywords_response, top_k=top_k_for_events)
   entities_result = entities_vdb.query(rewrite_entity_response, top_k=top_k_for_entities)
   ```

---

## Key Differences to Address

### 1. **Return Type Mismatch**
- **ICDCS**: Returns `List[Node]` objects with structured fields (id, type, score, embedding, content, metadata)
- **AVA**: Returns `List[Dict]` with flat structure and `__metrics__` for similarity score

### 2. **Query Interface**
- **ICDCS**: `search_events_by_description(query_text, top_k)` - text input
- **AVA**: `events_vdb.query(query_text, top_k)` - same interface, different implementation

### 3. **Object vs Entity Terminology**
- **ICDCS**: Uses "objects" (tracked objects from video)
- **AVA**: Uses "entities" (extracted entities from events)
- **Mapping**: ICDCS objects ≈ AVA entities (both represent tracked items in video)

### 4. **Metadata Structure**
- **ICDCS Events**: Has `description`, `objects` (comma-separated IDs), `start_time`, `end_time`
- **AVA Events**: Has `description`, `duration`, `id`
- **ICDCS Objects**: Has `track_id`, `class_name`, `frame_number`, `event_id`
- **AVA Entities**: Has `id`, `descriptions`, `events` (list), `timestamps`, `durations`

### 5. **Connection Management**
- **ICDCS**: Uses Milvus connection aliases, needs explicit disconnect
- **AVA**: Uses file-based NanoVectorDB, no connection management needed

---

## Step-by-Step Migration Plan

### **PHASE 1: Create Adapter Layer** ⚠️ CRITICAL FIRST STEP

**Goal**: Create a wrapper/adapter that allows KnowledgeGraphInterface to use AVA's vector databases while maintaining the same interface.

#### Step 1.1: Create AVA Database Adapter
**File**: `ICDCS/ava_db_adapter.py` (NEW FILE)

**Purpose**: Wrap AVA's BaseVectorStorage to provide MilvusDB-like interface

**Key Components**:
1. `AVAVectorDBAdapter` class that:
   - Wraps `BaseVectorStorage` (events_vdb or entities_vdb)
   - Implements methods compatible with current MilvusDB usage:
     - `search(query_embedding, k)` → Convert embedding to text query, call `vdb.query()`
     - `query(expr)` → Filter results by expression (may need custom implementation)
     - `get_by_id(id)` → Call `vdb.get_data(id)`
   - Converts AVA's Dict results to ICDCS's Node format

2. **Node Conversion Function**:
   ```python
   def ava_event_to_node(ava_result: dict) -> Node:
       # Convert AVA event dict to ICDCS Node
       # Extract embedding if available, or compute from description
   ```

#### Step 1.2: Update KnowledgeGraphInterface Initialization
**File**: `ICDCS/graph_interfaces.py`

**Changes**:
- Replace `MilvusDB` imports with AVA imports
- Modify `__init__` to accept AVA vector databases instead of file paths:
  ```python
  def __init__(self,
               events_vdb: BaseVectorStorage,  # Instead of event_faiss_db_path
               entities_vdb: BaseVectorStorage,  # Instead of object_faiss_db_path
               object_sqlite_db_path: str = "",  # Keep SQLite for structural queries
               embedding_model = None,
               embedding_dim: int = 768):
  ```
- Initialize adapters:
  ```python
  self.events_vdb = events_vdb  # Direct use or via adapter
  self.entities_vdb = entities_vdb  # Direct use or via adapter
  ```

---

### **PHASE 2: Update Query Methods**

#### Step 2.1: Update `search_events_by_description()`
**File**: `ICDCS/graph_interfaces.py` (Line 304)

**Current**:
```python
def search_events_by_description(self, description: str, top_k: int = 20) -> List[Node]:
    query_embedding = self.embedding_model.get_text_features([description])[0]
    return self.search_events_by_embedding(query_embedding, top_k)
```

**New**:
```python
def search_events_by_description(self, description: str, top_k: int = 20) -> List[Node]:
    # Use AVA's query method directly (it handles text-to-embedding internally)
    results = self.events_vdb.query(description, top_k=top_k)
    # Convert AVA results to Node objects
    return [self._ava_event_to_node(r) for r in results]
```

#### Step 2.2: Update `search_objects_by_description()`
**File**: `ICDCS/graph_interfaces.py` (Line 325)

**Current**:
```python
def search_objects_by_description(self, description: str, top_k: int = 20) -> List[Node]:
    query_embedding = self.embedding_model.get_text_features([description])[0]
    return self.search_objects_by_embedding(query_embedding, top_k)
```

**New**:
```python
def search_objects_by_description(self, description: str, top_k: int = 20) -> List[Node]:
    # Use AVA's query method for entities
    results = self.entities_vdb.query(description, top_k=top_k)
    # Convert AVA entity results to Node objects (treating entities as objects)
    return [self._ava_entity_to_node(r) for r in results]
```

#### Step 2.3: Update Embedding-based Search Methods
**File**: `ICDCS/graph_interfaces.py` (Lines 216, 258)

**Challenge**: AVA's `query()` takes text, not embeddings. Need to either:
- Option A: Keep embedding-based methods but convert embedding back to text (not ideal)
- Option B: Remove embedding-based methods and only use text-based
- Option C: Add embedding-to-text conversion (requires reverse embedding lookup)

**Recommendation**: Option B - Remove embedding-based methods or make them call text-based methods with a placeholder text.

#### Step 2.4: Create Conversion Helper Methods
**File**: `ICDCS/graph_interfaces.py`

**New Methods**:
```python
def _ava_event_to_node(self, ava_result: dict) -> Node:
    """Convert AVA event result to ICDCS Node"""
    # Extract embedding from description if needed
    # Map AVA fields to Node fields
    pass

def _ava_entity_to_node(self, ava_result: dict) -> Node:
    """Convert AVA entity result to ICDCS Node (treating as object)"""
    # Map entity to object structure
    pass
```

---

### **PHASE 3: Update Structural Query Methods**

#### Step 3.1: Update `get_objects_in_event()`
**File**: `ICDCS/graph_interfaces.py` (Line 347)

**Current**: Queries Milvus event DB by ID, extracts `objects` field (comma-separated string)

**New**: 
- Use `events_vdb.get_data(event_id)` to get event
- Check if event has `objects` field (may need to maintain compatibility)
- OR: Query entities_vdb to find entities that have this event_id in their `events` list

#### Step 3.2: Update `get_events_containing_object()`
**File**: `ICDCS/graph_interfaces.py` (Line 377)

**Current**: Queries Milvus object DB by track_id, gets frame_number, then finds events

**New**:
- Use `entities_vdb.get_data(entity_id)` to get entity
- Extract `events` field (list of event IDs) directly
- This is actually simpler with AVA's structure!

---

### **PHASE 4: Update Initialization in Benchmark Script**

#### Step 4.1: Update `run_ava100_benchmark.py`
**File**: `ICDCS/run_ava100_benchmark.py`

**Changes**:
1. **Remove Milvus imports** (Line 24):
   ```python
   # REMOVE: from pymilvus import connections
   ```

2. **Add AVA imports**:
   ```python
   from AVA.ava import AVA
   from AVA.base import BaseVectorStorage
   ```

3. **Update `get_or_create_graph_engine()`** (Lines 166-232):
   - Remove Milvus connection cleanup (Lines 195-200)
   - Instead of creating KnowledgeGraphInterface with file paths, create AVA instance and pass its vector databases:
     ```python
     # Create AVA instance (or reuse if same video)
     ava = AVA(video=video, llm_model=self.llm)
     ava.construct()  # Build knowledge graph if needed
     
     # Create KnowledgeGraphInterface with AVA's vector databases
     self.current_kg = KnowledgeGraphInterface(
         events_vdb=ava.events_vdb,
         entities_vdb=ava.entities_vdb,
         object_sqlite_db_path=db_paths['sqlite_db'],
         embedding_model=self.embedding_model,
         embedding_dim=768
     )
     ```

4. **Update `cleanup_graph_components()`** (Lines 146-164):
   - Remove Milvus connection disconnection code
   - Keep instance cleanup (set to None)

5. **Update `get_db_paths()`** (Lines 67-74):
   - May need to add video path or working directory for AVA
   - AVA uses `working_dir` instead of individual DB files

---

### **PHASE 5: Handle Video/Data Loading**

#### Step 5.1: AVA Video Initialization
**Challenge**: AVA requires a `VideoRepresentation` object, not just database paths.

**Solution Options**:
1. **Option A**: Load video in benchmark script and pass to AVA
2. **Option B**: Create AVA instance per video in benchmark
3. **Option C**: Reuse AVA's existing video loading mechanism

**Recommended**: Option B - Create AVA instance per video, similar to how it's done in `query_SA.py`

#### Step 5.2: Update Database Path Logic
**File**: `ICDCS/run_ava100_benchmark.py`

**Current**: `get_db_paths()` returns paths to Milvus DB files

**New**: Need to determine:
- Video path for AVA
- Working directory for AVA (where it stores vector DB files)
- May need to map video_key to video path

---

### **PHASE 6: Testing & Validation**

#### Step 6.1: Unit Tests
- Test `_ava_event_to_node()` conversion
- Test `_ava_entity_to_node()` conversion
- Test `search_events_by_description()` with AVA DB
- Test `search_objects_by_description()` with AVA DB
- Test `get_objects_in_event()` with AVA structure
- Test `get_events_containing_object()` with AVA structure

#### Step 6.2: Integration Tests
- Test single query through `graph_engine.py`
- Test full benchmark run with one video
- Compare results with Milvus version (if possible)

#### Step 6.3: Performance Validation
- Ensure query performance is acceptable
- Check memory usage (AVA uses file-based storage, may be different)

---

## Implementation Order (Recommended)

1. ✅ **Phase 1**: Create adapter/conversion layer (enables testing)
2. ✅ **Phase 2**: Update query methods (core functionality)
3. ✅ **Phase 3**: Update structural queries (completes interface)
4. ✅ **Phase 4**: Update benchmark script (enables end-to-end testing)
5. ✅ **Phase 5**: Handle video/data loading (completes integration)
6. ✅ **Phase 6**: Testing & validation (ensures correctness)

---

## Potential Challenges & Solutions

### Challenge 1: Embedding Availability
**Issue**: AVA's `query()` returns results but may not include embeddings in the result dict.

**Solution**: 
- Compute embedding on-demand from description using `embedding_model`
- Or modify conversion to extract embedding if available

### Challenge 2: Metadata Field Mismatch
**Issue**: AVA events/entities have different field names than ICDCS expects.

**Solution**: 
- Create mapping dictionaries in conversion functions
- Handle missing fields gracefully with defaults

### Challenge 3: SQLite Dependency
**Issue**: ICDCS still uses SQLite for some structural queries.

**Solution**: 
- Keep SQLite for now (it's separate from vector DB)
- May migrate to AVA's graph storage later if needed

### Challenge 4: Video Loading in Benchmark
**Issue**: Benchmark script currently only has database paths, not video objects.

**Solution**: 
- Add video loading logic to benchmark script
- Use dataset initialization similar to `query_SA.py`

### Challenge 5: Connection Management Removal
**Issue**: Removing Milvus connection management may affect cleanup logic.

**Solution**: 
- Simplify cleanup to just set instances to None
- AVA's file-based storage doesn't need connection cleanup

---

## Files to Modify

### Core Changes:
1. `ICDCS/graph_interfaces.py` - Main interface changes
2. `ICDCS/run_ava100_benchmark.py` - Initialization and connection management
3. `ICDCS/graph_engine.py` - May need minor adjustments if interface changes

### New Files:
1. `ICDCS/ava_db_adapter.py` - Optional adapter layer (if needed)

### Files to Review (May Need Changes):
1. `ICDCS/export_subgraph.py` - If it depends on Node structure
2. `ICDCS/graph_scorer.py` - If it depends on Node metadata

---

## Success Criteria

✅ ICDCS can query events using AVA's `events_vdb`  
✅ ICDCS can query entities using AVA's `entities_vdb`  
✅ Results are converted to ICDCS `Node` format correctly  
✅ Structural queries (`get_objects_in_event`, `get_events_containing_object`) work  
✅ Benchmark script runs end-to-end without Milvus  
✅ Query results are semantically equivalent to Milvus version  
✅ Performance is acceptable (may differ due to different storage backend)

---

## Next Steps

1. Review this plan and confirm approach
2. Start with Phase 1 (adapter layer) for testing
3. Iterate through phases, testing after each
4. Final integration testing with full benchmark

