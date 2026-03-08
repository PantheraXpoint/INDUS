# Project AVAS - Advanced Video Analysis and Search System

A comprehensive embedding-based video analysis system that combines object detection, tracking, event recognition, and semantic search capabilities using state-of-the-art AI models.

## Overview

Project AVAS (Advanced Video Analysis and Search) provides intelligent video understanding through:
- **Object Detection & Tracking**: YOLOv11-based detection with persistent tracking
- **Embedding Generation**: Semantic embeddings using JinaCLIP for similarity search
- **Event Recognition**: LLM-powered event description and tracking
- **Natural Language Search**: Query videos using text descriptions to find specific objects or events
- **Multi-Database Storage**: FAISS for fast similarity search and SQLite for detailed metadata

## Key Features

### 🎯 Object Detection & Tracking
- Real-time object detection using YOLOv11
- Persistent multi-object tracking across frames
- Automatic embedding generation for tracked objects
- Processing at 10 FPS for efficient tracking

### 🔍 Semantic Search
- Text-based object search using natural language descriptions
- Event-level search with contextual understanding
- Tri-view retrieval combining object and event embeddings
- Frame-range filtering for temporal queries

### 🤖 Event Recognition
- LLM-powered event description generation
- Chunk-based video analysis at 3 FPS
- Contextual event understanding with object relationships
- Event-level embedding storage for semantic search

### 💾 Database Systems
- **FAISS Database**: Fast similarity search for embeddings
- **SQLite Database**: Detailed tracking metadata (frames, bboxes, confidence)
- Automatic database management per video

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for FAISS-GPU and model inference)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For better performance, install `faiss-gpu` if you have CUDA:
```bash
pip install faiss-gpu
```

### Download Model Checkpoints

Ensure you have the YOLO model checkpoints in the `checkpoints/` directory:
- `checkpoints/yolo11l.pt` (default)
- `checkpoints/yolo11n.pt` (optional, lighter model)

## Quick Start

### 1. Process a Video

Process a video to generate embeddings and tracking data:

```bash
python main_embedding_tracking.py \
    --video path/to/your/video.mp4 \
    --process-only \
    --model qwenvl
```

This will:
- Detect and track objects in the video
- Generate embeddings for all tracked objects
- Generate event descriptions using LLM
- Store data in `database/<video_name>/` directory

### 2. Search for Objects

Search for specific objects or events in a processed video:

```bash
python main_embedding_tracking.py \
    --video path/to/your/video.mp4 \
    --description "a person wearing a red shirt" \
    --output-dir extracted_objects \
    --k 5 \
    --max-images 10 \
    --model qwenvl
```

This will:
- Parse your natural language query
- Search both object and event embeddings
- Filter and rank results using LLM
- Extract bounding box images from matching objects

### 3. View Database Statistics

Get statistics about tracked objects in a processed video:

```bash
python main_embedding_tracking.py \
    --video path/to/your/video.mp4 \
    --stats
```

## Usage Examples

### Example 1: Process and Search
```bash
# Step 1: Process the video
python main_embedding_tracking.py --video sample.mp4 --process-only

# Step 2: Search for objects
python main_embedding_tracking.py \
    --video sample.mp4 \
    --description "a person with blue shirt" \
    --k 10
```

### Example 2: Complex Query Search
```bash
# Search with natural language query
python main_embedding_tracking.py \
    --video sample.mp4 \
    --description "find a child playing in the playground" \
    --output-dir results \
    --max-images 5
```

### Example 3: Custom Model
```bash
# Use a different LLM model
python main_embedding_tracking.py \
    --video sample.mp4 \
    --description "a car in the parking lot" \
    --model qwenlm
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--video` | Path to input video file (required) | - |
| `--model` | LLM model to use (qwenvl, qwenlm, etc.) | `qwenvl` |
| `--description` | Text description to search for | - |
| `--output-dir` | Directory to save extracted images | `extracted_objects` |
| `--k` | Number of search results to return | `5` |
| `--max-images` | Maximum images to extract per track | `1` |
| `--process-only` | Only process video without searching | `False` |
| `--stats` | Show database statistics | `False` |

## Architecture

### Processing Pipeline

```
Video Input
    ↓
[Object Detection] → YOLOv11 (10 FPS)
    ↓
[Object Tracking] → Persistent IDs
    ↓
[Embedding Generation] → JinaCLIP embeddings
    ↓
[Database Storage] → FAISS + SQLite
    ↓
[Event Tracking] → LLM event descriptions (3 FPS)
    ↓
[Event Embeddings] → FAISS storage
```

### Search Pipeline

```
Natural Language Query
    ↓
[LLM Query Parsing] → Extract intent & constraints
    ↓
[Tri-View Retrieval] → Event + Object search
    ↓
[Result Filtering] → LLM-based ranking
    ↓
[Image Extraction] → Bounding box images
```

## Project Structure

```
Project-AVAS/
├── main_embedding_tracking.py    # Main script
├── AVA/                           # Core analysis modules
│   ├── object_detect.py          # Object detection & tracking
│   ├── event_tracker.py          # Event recognition
│   └── utils.py                  # Utility functions
├── embeddings/                    # Embedding system
│   ├── JinaCLIP.py               # JinaCLIP wrapper
│   ├── FAISSDB.py                # FAISS database
│   ├── SQLiteDB.py               # SQLite database
│   └── object_search.py          # Search system
├── llms/                         # Language models
│   ├── init_model.py             # Model initialization
│   ├── QwenVL.py                 # QwenVL model
│   └── QwenLM.py                 # QwenLM model
├── checkpoints/                   # Model checkpoints
│   └── yolo11l.pt
├── config/                        # Configuration files
│   └── tracker.yaml
├── database/                      # Processed video databases
│   └── <video_name>/
│       ├── object_embeddings.faiss
│       ├── event_embeddings.faiss
│       └── tracked_objects.db
└── extracted_objects/            # Extracted images
```

## Database Schema

### SQLite Database (`tracked_objects.db`)

**tracked_objects table:**
- `track_id`: Unique track identifier
- `class_id`: Object class ID
- `class_name`: Object class name
- `first_frame`: First appearance frame
- `last_frame`: Last appearance frame
- `total_frames`: Total tracked frames
- `bbox_history`: JSON array of bounding boxes
- `confidence_history`: JSON array of confidence scores
- `frame_numbers`: JSON array of frame numbers

**video_info table:**
- `video_path`: Path to video file
- `width`: Video width
- `height`: Video height
- `fps`: Original video FPS
- `total_frames`: Total frame count
- `processing_fps`: Processing FPS used

### FAISS Databases

- **object_embeddings.faiss**: Object-level embeddings for similarity search
- **event_embeddings.faiss**: Event-level embeddings for event search

## API Usage

### Process Video Programmatically

```python
from main_embedding_tracking import process_video

# Process video and generate embeddings
result = process_video(
    video_path="path/to/video.mp4",
    object_faiss_db_path="database/video/object_embeddings.faiss",
    event_faiss_db_path="database/video/event_embeddings.faiss",
    object_sqlite_db_path="database/video/tracked_objects.db",
    model="qwenvl"
)
```

### Search Programmatically

```python
from main_embedding_tracking import search_video
from llms.init_model import init_model

# Initialize LLM
llm = init_model("qwenvl", 1)

# Search for objects
search_results, saved_images = search_video(
    query="a person wearing red shirt",
    video_path="path/to/video.mp4",
    output_dir="extracted_objects",
    object_faiss_db_path="database/video/object_embeddings.faiss",
    event_faiss_db_path="database/video/event_embeddings.faiss",
    object_sqlite_db_path="database/video/tracked_objects.db",
    k=5,
    max_images=10,
    llm=llm
)
```

## Performance Notes

- **Processing Speed**: ~10 FPS for tracking, ~3 FPS for event generation
- **Memory Usage**: FAISS database is memory-efficient for large-scale search
- **GPU Acceleration**: Recommended for FAISS-GPU and model inference
- **Storage**: Each video creates its own database directory

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure model checkpoints are in `checkpoints/` directory
2. **CUDA Out of Memory**: 
   - Use CPU version: `pip install faiss-cpu`
   - Reduce batch size in model configuration
3. **FAISS Import Error**: Install appropriate version:
   ```bash
   pip install faiss-gpu  # For CUDA
   # or
   pip install faiss-cpu  # For CPU
   ```
4. **Database Not Found**: Process video first using `--process-only` flag

### Performance Optimization

1. Use `faiss-gpu` for faster similarity search on CUDA systems
2. Adjust processing FPS in code for balance between speed and accuracy
3. Use lighter YOLO models (`yolo11n.pt`) for faster processing

## Advanced Features

### Tri-View Retrieval

The system uses a sophisticated retrieval method that combines:
- **Event Search**: High-level event descriptions
- **Object Search**: Fine-grained object matching
- **LLM Filtering**: Intelligent result ranking and filtering

### Event Tracking

Events are generated by analyzing video chunks:
- Chunk duration: 3 seconds
- Processing rate: 3 FPS
- Contextual understanding of object interactions

## License

See `LICENSE` file for details.

## References

- **YOLOv11**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **JinaCLIP**: [Jina AI](https://jina.ai/)
- **FAISS**: [Facebook AI Research](https://github.com/facebookresearch/faiss)
- **QwenVL/QwenLM**: [Alibaba Cloud](https://github.com/QwenLM/Qwen2-VL)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Support

For issues and questions, please open an issue on the project repository.

