"""
Pre-computed Exploit Vector Library
Manage and access pre-generated exploit vectors
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ExploitLibrary:
    """Manage library of pre-computed exploit vectors"""
    
    def __init__(self, db_path: str = "data/exploits/exploit_library.db"):
        """
        Initialize exploit library
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema if needed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exploits (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                trait TEXT,
                objective TEXT,
                payload TEXT NOT NULL,
                vector_norm REAL,
                success_rate REAL,
                stealth_score REAL,
                complexity INTEGER,
                created_at TIMESTAMP,
                tested_models TEXT,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_success 
            ON exploits(success_rate DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category 
            ON exploits(category)
        """)
        
        conn.commit()
        conn.close()
    
    def add_exploit(self, exploit_dict: Dict[str, Any]) -> bool:
        """
        Add exploit to library
        
        Args:
            exploit_dict: Exploit data dictionary
            
        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO exploits 
                (id, category, trait, objective, payload, vector_norm,
                 success_rate, stealth_score, complexity, created_at,
                 tested_models, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exploit_dict.get("id"),
                exploit_dict.get("category"),
                exploit_dict.get("trait"),
                exploit_dict.get("objective"),
                exploit_dict.get("payload"),
                exploit_dict.get("vector_norm", 0),
                exploit_dict.get("success_rate", 0),
                exploit_dict.get("stealth_score", 0),
                exploit_dict.get("complexity", 5),
                exploit_dict.get("created_at"),
                json.dumps(exploit_dict.get("tested_models", [])),
                json.dumps(exploit_dict.get("tags", [])),
                json.dumps(exploit_dict.get("metadata", {}))
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add exploit: {e}")
            return False
        finally:
            conn.close()
    
    def get_exploit(self, exploit_id: str) -> Optional[Dict[str, Any]]:
        """
        Get exploit by ID
        
        Args:
            exploit_id: Exploit ID
            
        Returns:
            Exploit dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM exploits WHERE id = ?", (exploit_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def search(
        self,
        category: Optional[str] = None,
        trait: Optional[str] = None,
        min_success_rate: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search exploits by criteria
        
        Args:
            category: Filter by category
            trait: Filter by trait
            min_success_rate: Minimum success rate
            limit: Maximum results
            
        Returns:
            List of exploit dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM exploits WHERE success_rate >= ?"
        params = [min_success_rate]
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if trait:
            query += " AND trait = ?"
            params.append(trait)
        
        query += " ORDER BY success_rate DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_top_exploits(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N exploits by success rate
        
        Args:
            n: Number of exploits
            
        Returns:
            List of exploit dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM exploits 
            ORDER BY success_rate DESC 
            LIMIT ?
        """, (n,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM exploits")
        total = cursor.fetchone()[0]
        
        # Category distribution
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM exploits 
            GROUP BY category
        """)
        categories = dict(cursor.fetchall())
        
        # Success rate stats
        cursor.execute("""
            SELECT 
                AVG(success_rate) as avg_success,
                MIN(success_rate) as min_success,
                MAX(success_rate) as max_success
            FROM exploits
        """)
        success_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_exploits": total,
            "categories": categories,
            "average_success_rate": success_stats[0] or 0,
            "min_success_rate": success_stats[1] or 0,
            "max_success_rate": success_stats[2] or 0
        }
    
    def export_json(self, output_path: str) -> int:
        """
        Export library to JSON
        
        Args:
            output_path: Output file path
            
        Returns:
            Number of exploits exported
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM exploits ORDER BY success_rate DESC")
        rows = cursor.fetchall()
        conn.close()
        
        exploits = [self._row_to_dict(row) for row in rows]
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                "version": "1.0",
                "total": len(exploits),
                "exploits": exploits
            }, f, indent=2, default=str)
        
        logger.info(f"Exported {len(exploits)} exploits to {output_path}")
        return len(exploits)
    
    def export_csv(self, output_path: str) -> int:
        """
        Export library to CSV
        
        Args:
            output_path: Output file path
            
        Returns:
            Number of exploits exported
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM exploits", conn)
        conn.close()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(df)} exploits to {output_path}")
        return len(df)
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        return {
            "id": row[0],
            "category": row[1],
            "trait": row[2],
            "objective": row[3],
            "payload": row[4],
            "vector_norm": row[5],
            "success_rate": row[6],
            "stealth_score": row[7],
            "complexity": row[8],
            "created_at": row[9],
            "tested_models": json.loads(row[10]) if row[10] else [],
            "tags": json.loads(row[11]) if row[11] else [],
            "metadata": json.loads(row[12]) if row[12] else {}
        }