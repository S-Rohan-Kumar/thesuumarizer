from flask import Flask, jsonify, request,render_template
import sqlite3
import time
import os
import requests
import json
from screenpipe_launcher import launch_screenpipe_in_terminal
from main import extract_content_for_ai, get_recent_frames, capture_new_content, send_to_deepseek, save_content_to_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_PATH = fr"C:\Users\{os.getlogin()}\.screenpipe\db.sqlite"
@app.route('/screenpipe')
def home():
    return render_template("index.html")
@app.route('/api/launch_screenpipe', methods=['GET'])
def launch_screenpipe():
    success = launch_screenpipe_in_terminal()
    return jsonify({"status": "success" if success else "error", "message": "Screenpipe launched" if success else "Failed to launch Screenpipe"})

@app.route('/api/frames', methods=['GET'])
def list_frames():
    limit = request.args.get('limit', default=10, type=int)
    frames = get_recent_frames(limit)
    return jsonify(frames)

@app.route('/api/capture', methods=['POST'])
def capture_content():
    content_data = capture_new_content()
    if content_data:
        return jsonify(content_data)
    return jsonify({"status": "error", "message": "No content captured"}), 400

@app.route('/api/process/<int:frame_id>', methods=['POST'])
def process_frame(frame_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First check if created_at column exists
    cursor.execute("PRAGMA table_info(ocr_text)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Build query based on available columns
    if "created_at" in columns:
        cursor.execute("SELECT frame_id, text, created_at FROM ocr_text WHERE frame_id = ?", (frame_id,))
    else:
        cursor.execute("SELECT frame_id, text FROM ocr_text WHERE frame_id = ?", (frame_id,))
    
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"status": "error", "message": "Frame not found"}), 404

    text = row[1]
    # Use created_at if available, otherwise use current time
    timestamp = row[2] if len(row) > 2 and row[2] else time.strftime("%Y-%m-%d %H:%M:%S")
    app_type, content = extract_content_for_ai(text, debug_mode=True)
    content_data = {
        "source_app": app_type or "Unknown",
        "content": content or text,
        "timestamp": timestamp,
        "frame_id": frame_id
    }

    action = request.json.get('action', 'analyze')
    result = send_to_deepseek(content_data, action)
    return jsonify({"status": "success", "result": result})

@app.route('/api/save/<int:frame_id>', methods=['POST'])
def save_frame(frame_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First check if created_at column exists
    cursor.execute("PRAGMA table_info(ocr_text)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Build query based on available columns
    if "created_at" in columns:
        cursor.execute("SELECT frame_id, text, created_at FROM ocr_text WHERE frame_id = ?", (frame_id,))
    else:
        cursor.execute("SELECT frame_id, text FROM ocr_text WHERE frame_id = ?", (frame_id,))
    
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"status": "error", "message": "Frame not found"}), 404

    text = row[1]
    # Use created_at if available, otherwise use current time
    timestamp = row[2] if len(row) > 2 and row[2] else time.strftime("%Y-%m-%d %H:%M:%S")
    app_type, content = extract_content_for_ai(text)
    content_data = {
        "source_app": app_type or "Unknown",
        "content": content or text,
        "timestamp": timestamp,
        "frame_id": frame_id
    }

    if save_content_to_file(content_data):
        return jsonify({"status": "success", "message": f"Content saved for frame {frame_id}"})
    return jsonify({"status": "error", "message": "Failed to save content"}), 500

if __name__ == "__main__":
    launch_screenpipe_in_terminal()
    app.run(debug=True,  port=7000)