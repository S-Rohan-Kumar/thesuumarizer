import sqlite3
import time
import re
from datetime import datetime
import os
import requests
import json
import screenpipe_launcher

# Configuration
username = os.getlogin()
DB_PATH = fr"C:\Users\{username}\.screenpipe\db.sqlite"
WAIT_TIME = 5
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def extract_content_for_ai(ocr_text, app_type=None, debug_mode=False):
    """
    Extract meaningful content from OCR text by filtering out UI elements
    based on the detected application type.
    """
    if not ocr_text or not isinstance(ocr_text, str):
        return None, None
        
    # Only detect app_type if not already provided
    if app_type is None:
        if any(x in ocr_text for x in ["Notepad", "File Edit", "Ln", "Col"]):
            app_type = "Notepad"
        elif any(x in ocr_text for x in ["Chrome", "http://", "https://", "www.", ".com"]):
            app_type = "Chrome"
        elif any(x in ocr_text for x in ["Word", "Document", ".docx", "Microsoft Word"]):
            app_type = "Word"
        else:
            app_type = "Unknown"
    
    if debug_mode:
        print(f"Detected app type: {app_type}")
        print(f"Original OCR text length: {len(ocr_text)} characters")
    
    lines = ocr_text.split('\n')
    content_lines = []
    filtered_lines = []
    
    if app_type == "Notepad":
        for line in lines:
            ui_patterns = ["File Edit", "Ln", "Col", "UTF-8", "Windows (CRLF)", "- Notepad"]
            if debug_mode and any(ui in line for ui in ui_patterns):
                filtered_lines.append(f"[UI-FILTERED] {line}")
                continue
            elif not debug_mode and any(ui in line for ui in ui_patterns):
                continue
            if re.search(r'PS C:\\|>', line) and not debug_mode:
                continue
            if line.strip():
                content_lines.append(line)
                
    elif app_type == "Chrome":
        for line in lines:
            ui_patterns = ["http://", "https://", "New Tab", "Settings", "- Google Chrome"]
            if debug_mode and any(ui in line for ui in ui_patterns):
                filtered_lines.append(f"[UI-FILTERED] {line}")
                continue
            elif not debug_mode and any(ui in line for ui in ui_patterns):
                continue
            if not debug_mode and re.search(r'(File|Edit|View|History|Bookmarks|People|More)', line):
                continue
            if line.strip():
                content_lines.append(line)
                
    elif app_type == "Word":
        for line in lines:
            ui_patterns = ["File", "Home", "Insert", "Design", "Layout", "Review", "View", "Microsoft Word"]
            if debug_mode and any(ui in line for ui in ui_patterns):
                filtered_lines.append(f"[UI-FILTERED] {line}")
                continue
            elif not debug_mode and any(ui in line for ui in ui_patterns):
                continue
            if line.strip():
                content_lines.append(line)
    else:
        for line in lines:
            if not debug_mode and re.search(r'(File|Edit|View|Window|Help|Tools|Options)', line):
                continue
            if not debug_mode and len(line) < 15 and re.search(r'[☰☑⋮→←↑↓⚙️🔍]', line):
                continue
            if line.strip():
                content_lines.append(line)
    
    content = '\n'.join(content_lines).strip()
    if not debug_mode:
        content = re.sub(r'^\s*(?:x )?File\s+', '', content)
        content = re.sub(r'\b\w{1,3}\b\s*\n', '\n', content)
    
    if debug_mode:
        print(f"Content after filtering: {len(content)} characters")
        if len(content) < 10 and len(ocr_text) > 10:
            print("WARNING: Content too small after filtering. Using original OCR text.")
            content = ocr_text
    
    return app_type, content

def get_recent_frames(limit=10):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(ocr_text)")
        columns = [col[1] for col in cursor.fetchall()]
        fields = ["frame_id", "text"]
        if "created_at" in columns:
            fields.append("created_at")
        
        query = f"SELECT {', '.join(fields)} FROM ocr_text WHERE text IS NOT NULL ORDER BY frame_id DESC LIMIT ?"
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        frames = []
        for row in rows:
            frame_id = row[0]
            text = row[1]
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            timestamp = row[2] if len(row) > 2 and row[2] else current_time
            app_type, _ = extract_content_for_ai(text)
            preview = text[:50] + "..." if len(text) > 50 else text
            frames.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "app_type": app_type or "Unknown",
                "preview": preview,
                "full_text": text
            })
        return frames
    except Exception as e:
        print(f"Error retrieving frames: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def capture_new_content(wait_time=WAIT_TIME):
    print(f"Capturing new screen content in:")
    for i in range(wait_time, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)
    print("Capturing!")
    time.sleep(2)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(ocr_text)")
        columns = [col[1] for col in cursor.fetchall()]
        fields = ["text", "frame_id"]
        if "created_at" in columns:
            fields.append("created_at")
        
        query = f"SELECT {', '.join(fields)} FROM ocr_text WHERE text IS NOT NULL ORDER BY frame_id DESC LIMIT 1"
        cursor.execute(query)
        row = cursor.fetchone()
        
        if row:
            text = row[0]
            frame_id = row[1]
            timestamp = row[2] if len(row) > 2 and row[2] else time.strftime("%Y-%m-%d %H:%M:%S")
            app_type, content = extract_content_for_ai(text, debug_mode=True)
            
            if not content or len(content) < 10:
                print("\nWARNING: Very little meaningful content detected.")
                content = text
            
            content_data = {
                "source_app": app_type or "Unknown",
                "content": content or text,
                "timestamp": timestamp,
                "frame_id": frame_id
            }
            print(f"\n✓ CAPTURED CONTENT FROM {app_type} (frame_id: {frame_id}):")
            print(f"Preview: {content_data['content'][:100]}...")
            return content_data
        
        print("No content detected in recent screen capture.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def send_to_deepseek(content_data, action="analyze"):
    if not content_data:
        return "No content to process"
    
    print(f"Sending {len(content_data['content'])} characters to DeepSeek AI...")
    
    if action == "summarize":
        system_prompt = "You are an AI assistant that summarizes content from screen captures."
        user_prompt = f"Summarize the following content from {content_data['source_app']}:\n\n{content_data['content']}"
    elif action == "explain":
        system_prompt = "You are an AI assistant that explains complex content in simple terms."
        user_prompt = f"Explain the following content from {content_data['source_app']} in clear, simple terms:\n\n{content_data['content']}"
    elif action == "code_help":
        system_prompt = "You are an expert programmer helping to understand and improve code."
        user_prompt = f"Analyze this code and provide insights, improvements and explanations:\n\n{content_data['content']}"
    else:
        system_prompt = "You are an AI assistant that analyzes content from screen captures."
        user_prompt = f"Analyze the following content from {content_data['source_app']} and provide insights:\n\n{content_data['content']}"
    
    try:
        payload = {
            "model": "deepseek/deepseek-r1-distill-llama-70b:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        print("Sending request to DeepSeek API...")
        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            if ai_response:
                os.makedirs('saved_responses', exist_ok=True)
                save_path = os.path.join('saved_responses', f"deepseek_response_{int(time.time())}.txt")
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source: {content_data['source_app']}\n")
                    f.write(f"Action: {action}\n")
                    f.write(f"Timestamp: {content_data['timestamp']}\n\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(ai_response)
                print(f"DeepSeek response saved to {save_path}")
                return ai_response
            return "Empty response received from DeepSeek API."
        return f"Error from DeepSeek API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error sending to DeepSeek: {str(e)}"

def save_content_to_file(content_data):
    if not content_data:
        return False
        
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    app_name = content_data['source_app'].lower().replace(' ', '_')
    filename = f"{app_name}_content_{timestamp}.txt"
    
    try:
        os.makedirs('saved_content', exist_ok=True)
        filepath = os.path.join('saved_content', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Source Application: {content_data['source_app']}\n")
            f.write(f"Timestamp: {content_data['timestamp']}\n")
            f.write(f"Frame ID: {content_data['frame_id']}\n\n")
            f.write("=" * 50 + "\n\n")
            f.write(content_data['content'])
        print(f"Content saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving content: {e}")
        return False

def deepseek_menu(content_data):
    print("\n==== DeepSeek AI Processing Options ====")
    print("1. Analyze content")
    print("2. Summarize content")
    print("3. Explain content in simple terms")
    print("4. Code help")
    print("5. Return to previous menu")
    
    choice = input("\nSelect an option (1-5): ").strip()
    if choice == '1':
        return send_to_deepseek(content_data, "analyze")
    elif choice == '2':
        return send_to_deepseek(content_data, "summarize")
    elif choice == '3':
        return send_to_deepseek(content_data, "explain")
    elif choice == '4':
        return send_to_deepseek(content_data, "code_help")
    elif choice == '5':
        return None
    else:
        print("Invalid option.")
        return None

def main_menu():
    while True:
        print("\n==== Screen Content Extractor with DeepSeek AI ====")
        print("1. View recent frames and select one")
        print("2. Capture new screen content")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        if choice == '1':
            frames = get_recent_frames(10)
            content_data = display_frame_selection_menu(frames)
            if content_data == "new_capture":
                content_data = capture_new_content()
            if content_data:
                process_selected_content(content_data)
        elif choice == '2':
            content_data = capture_new_content()
            if content_data:
                process_selected_content(content_data)
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid option.")

def display_frame_selection_menu(frames):
    if not frames:
        print("No recent frames available.")
        return None
        
    print("\n===== Recent Screen Captures =====")
    for i, frame in enumerate(frames):
        print(f"{i+1}. [{frame['timestamp']}] {frame['app_type']}: {frame['preview']}")
    
    print("\n0. Capture new screen content")
    print("Q. Quit")
    
    while True:
        choice = input("\nSelect a frame number, 0 for new capture, or Q to quit: ").strip().lower()
        if choice == 'q':
            return None
        try:
            choice_num = int(choice)
            if choice_num == 0:
                return "new_capture"
            if 1 <= choice_num <= len(frames):
                selected_frame = frames[choice_num-1]
                app_type, content = extract_content_for_ai(selected_frame["full_text"], debug_mode=True)
                print(f"\nDIAGNOSTIC INFO FOR FRAME {selected_frame['frame_id']}:")
                print(f"Application type: {app_type}")
                print(f"Original text length: {len(selected_frame['full_text'])} characters")
                print(f"Processed content length: {len(content) if content else 0} characters")
                if content:
                    content_data = {
                        "source_app": app_type,
                        "content": content,
                        "timestamp": selected_frame["timestamp"],
                        "frame_id": selected_frame["frame_id"]
                    }
                    print("\nContent after processing:")
                    print(f"---Preview (50 chars)---\n{content[:50]}...")
                    use_original = input("Use processed content? (Y/n): ").strip().lower()
                    if use_original == 'n':
                        content_data["content"] = selected_frame["full_text"]
                        print("Using original unprocessed text.")
                    return content_data
                else:
                    print("Warning: Selected frame doesn't contain content.")
                    use_original = input("Use original text? (Y/n): ").strip().lower()
                    if use_original != 'n':
                        content_data = {
                            "source_app": app_type or "Unknown",
                            "content": selected_frame["full_text"],
                            "timestamp": selected_frame["timestamp"],
                            "frame_id": selected_frame["frame_id"]
                        }
                        return content_data
                    print("No content to process.")
                    return None
            else:
                print(f"Please enter a number between 0 and {len(frames)}.")
        except ValueError:
            print("Please enter a valid number or 'Q'.")

def process_selected_content(content_data):
    if not content_data:
        return
    print("\n==== Selected Content ====")
    print(f"Source: {content_data['source_app']}")
    print(f"Time: {content_data['timestamp']}")
    print(f"Size: {len(content_data['content'])} characters")
    print(f"Preview: {content_data['content'][:100]}...")
    print("\nWhat would you like to do with this content?")
    print("1. Save to file")
    print("2. Process with DeepSeek AI")
    print("3. Save AND process with DeepSeek AI")
    print("4. Return to main menu")
    
    choice = input("\nSelect an option (1-4): ").strip()
    if choice == '1':
        save_content_to_file(content_data)
    elif choice == '2':
        result = deepseek_menu(content_data)
        if result:
            print("\nDEEPSEEK AI RESULT:")
            print("=" * 50)
            print(result)
            print("=" * 50)
    elif choice == '3':
        save_content_to_file(content_data)
        result = deepseek_menu(content_data)
        if result:
            print("\nDEEPSEEK AI RESULT:")
            print("=" * 50)
            print(result)
            print("=" * 50)
    elif choice == '4':
        return
    else:
        print("Invalid option.")

if __name__ == "__main__":
    print("Screen Content Extractor with DeepSeek AI")
    screenpipe_launcher.launch_screenpipe_in_terminal()
    main_menu()