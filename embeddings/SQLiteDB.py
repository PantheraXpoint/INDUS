import sqlite3
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class SQLiteDB:
    """
    SQLite database for storing tracked objects information
    """
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn = None
        self._create_tables()
        self._enable_wal_mode()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tracked_objects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracked_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER UNIQUE NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL,
                total_frames INTEGER NOT NULL,
                bbox_history TEXT NOT NULL,  -- JSON string of bbox coordinates
                confidence_history TEXT NOT NULL,  -- JSON string of confidence scores
                frame_numbers TEXT NOT NULL,  -- JSON string of frame numbers
                event_id TEXT NOT NULL,  -- JSON string of event IDs
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_id ON tracked_objects(track_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_class ON tracked_objects(track_id, class_id)')
        
        # Create video_info table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT UNIQUE NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                fps REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                processing_fps REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _enable_wal_mode(self):
        """Enable WAL mode for better concurrency and performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA cache_size=10000')
            conn.execute('PRAGMA temp_store=MEMORY')
            conn.close()
        except Exception as e:
            print(f"Warning: Could not enable WAL mode: {e}")
    
    def _get_connection(self):
        """Get or create a persistent connection"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute('PRAGMA journal_mode=WAL')
            self._conn.execute('PRAGMA synchronous=NORMAL')
            self._conn.execute('PRAGMA cache_size=10000')
        return self._conn
    
    def _reconnect(self):
        """Reconnect to the database"""
        try:
            if self._conn is not None:
                self._conn.close()
        except:
            pass
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA synchronous=NORMAL')
        self._conn.execute('PRAGMA cache_size=10000')
    
    def close(self):
        """Close the database connection"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def add_tracked_object(self, track_id: int, class_id: int, class_name: str, 
                          bbox_history: List[int], confidence_history: float, 
                          frame_numbers: int, event_id: int) -> bool:
        """
        Add or update tracked object information
        
        Args:
            track_id: Unique track ID
            class_id: Object class ID
            class_name: Object class name
            bbox_history: bounding box coordinates
            confidence_history: confidence scores
            frame_numbers: frame number where object was detected
            event_id: event ID where object was detected
        Returns:
            True if existing, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Single query to get all needed data (combines both previous SELECTs)
            cursor.execute('''
                SELECT bbox_history, confidence_history, frame_numbers, event_id 
                FROM tracked_objects 
                WHERE track_id = ?
            ''', (track_id,))
            existing_histories = cursor.fetchone()
            
            if existing_histories:
                # Update existing record, append new history to existing JSON lists
                # Load existing histories
                existing_bbox_history = json.loads(existing_histories[0])
                existing_confidence_history = json.loads(existing_histories[1])
                existing_frame_numbers = json.loads(existing_histories[2])
                existing_event_id = json.loads(existing_histories[3])

                # Append new histories
                updated_bbox_history = existing_bbox_history + [bbox_history]
                updated_confidence_history = existing_confidence_history + [confidence_history]
                updated_frame_numbers = existing_frame_numbers + [frame_numbers]
                updated_event_id = existing_event_id
                if event_id not in existing_event_id:
                    updated_event_id = existing_event_id + [event_id]

                # Serialize JSON once
                bbox_json = json.dumps(updated_bbox_history)
                conf_json = json.dumps(updated_confidence_history)
                frames_json = json.dumps(updated_frame_numbers)
                event_json = json.dumps(updated_event_id)
                
                max_frame = max(updated_frame_numbers)
                total_frames = len(updated_frame_numbers)

                cursor.execute('''
                    UPDATE tracked_objects 
                    SET class_id = ?, class_name = ?, last_frame = ?, total_frames = ?,
                        bbox_history = ?, confidence_history = ?, frame_numbers = ?,
                        event_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE track_id = ?
                ''', (class_id, class_name, max_frame, total_frames,
                      bbox_json, conf_json, frames_json, event_json, track_id))
                
                conn.commit()
                return True
            else:
                # Insert new record
                bbox_list = [bbox_history]
                conf_list = [confidence_history]
                frames_list = [frame_numbers]
                event_list = [event_id]
                
                # Serialize JSON once
                bbox_json = json.dumps(bbox_list)
                conf_json = json.dumps(conf_list)
                frames_json = json.dumps(frames_list)
                event_json = json.dumps(event_list)
                
                cursor.execute('''
                    INSERT INTO tracked_objects 
                    (track_id, class_id, class_name, first_frame, last_frame, total_frames,
                     bbox_history, confidence_history, frame_numbers, event_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (track_id, class_id, class_name, frame_numbers, frame_numbers, 1,
                      bbox_json, conf_json, frames_json, event_json))
                
                conn.commit()
                return False
            
        except (sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
            # Connection might be stale, try reconnecting once
            try:
                self._reconnect()
                conn = self._get_connection()
                cursor = conn.cursor()
                # Retry the operation (simplified - just return False on retry failure)
                print(f"Error adding tracked object, reconnected: {e}")
                return False
            except Exception as e2:
                print(f"Error adding tracked object after reconnect: {e2}")
                return False
        except Exception as e:
            print(f"Error adding tracked object: {e}")
            try:
                conn.rollback()
            except:
                pass
            return False
    
    def get_tracked_object(self, track_id: int) -> Optional[Dict]:
        """
        Get tracked object information by track_id
        
        Args:
            track_id: Track ID to search for
            
        Returns:
            Dictionary with object information or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM tracked_objects WHERE track_id = ?
            ''', (track_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'track_id': row[1],
                    'class_id': row[2],
                    'class_name': row[3],
                    'first_frame': row[4],
                    'last_frame': row[5],
                    'total_frames': row[6],
                    'bbox_history': json.loads(row[7]),
                    'confidence_history': json.loads(row[8]),
                    'frame_numbers': json.loads(row[9]),
                    'event_id': json.loads(row[10]),
                    'created_at': row[11],
                    'updated_at': row[12]
                }
            return None
            
        except Exception as e:
            print(f"Error getting tracked object: {e}")
            return None
    
    def get_all_tracked_objects(self) -> List[Dict]:
        """
        Get all tracked objects
        
        Returns:
            List of dictionaries with object information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tracked_objects ORDER BY track_id')
            rows = cursor.fetchall()
            conn.close()
            
            objects = []
            for row in rows:
                objects.append({
                    'id': row[0],
                    'track_id': row[1],
                    'class_id': row[2],
                    'class_name': row[3],
                    'first_frame': row[4],
                    'last_frame': row[5],
                    'total_frames': row[6],
                    'bbox_history': json.loads(row[7]),
                    'confidence_history': json.loads(row[8]),
                    'frame_numbers': json.loads(row[9]),
                    'created_at': row[10],
                    'updated_at': row[11]
                })
            
            return objects
            
        except Exception as e:
            print(f"Error getting all tracked objects: {e}")
            return []
    
    def add_video_info(self, video_path: str, width: int, height: int, 
                      fps: float, total_frames: int, processing_fps: float) -> bool:
        """
        Add video information
        
        Args:
            video_path: Path to video file
            width: Video width
            height: Video height
            fps: Video FPS
            total_frames: Total number of frames
            processing_fps: Processing FPS used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO video_info 
                (video_path, width, height, fps, total_frames, processing_fps)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (video_path, width, height, fps, total_frames, processing_fps))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error adding video info: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM video_info WHERE video_path = ?', (video_path,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'video_path': row[1],
                    'width': row[2],
                    'height': row[3],
                    'fps': row[4],
                    'total_frames': row[5],
                    'processing_fps': row[6],
                    'created_at': row[7]
                }
            return None
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None
