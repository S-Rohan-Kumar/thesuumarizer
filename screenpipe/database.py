import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = 'screen_extractor.db'

def initialize_database():
    """Create and initialize the database with required tables"""
    try:
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(DB_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create screenshots table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS screenshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_text TEXT,
            summary TEXT,
            processed INTEGER DEFAULT 0
        )
        ''')
        
        # Create a settings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        ''')
        
        # Insert default settings
        default_settings = [
            ('api_key', ''),
            ('output_dir', os.path.join(os.path.expanduser('~'), 'ScreenExtractor')),
            ('auto_summarize', '0'),
        ]
        
        for key, value in default_settings:
            cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def get_setting(key):
    """Get a setting value from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting setting {key}: {e}")
        return None

def save_setting(key, value):
    """Save a setting to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving setting {key}: {e}")
        return False

def save_screenshot(filename, raw_text=None, summary=None):
    """Save screenshot details to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO screenshots (filename, raw_text, summary) VALUES (?, ?, ?)",
            (filename, raw_text, summary)
        )
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return last_id
    except Exception as e:
        logger.error(f"Error saving screenshot: {e}")
        return None