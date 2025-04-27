from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from PIL import Image
import pytesseract
from groq import Groq
import moviepy as moviepy
import speech_recognition as sr
from pydub import AudioSegment
import os
import tempfile
import sys
from langi import extract_text
import yt_dlp
from pydub.utils import make_chunks
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import pdfplumber
import subprocess
import bcrypt
import signal
import time
import re
import numpy as np
import cv2
import logging
from dotenv import load_dotenv
import time
import requests
import json
# Load environment variables from .env file (for local development)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, supports_credentials=True)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key_for_dev')

# Database configuration from the email screenshot
app.config['MYSQL_HOST'] = 'sql12.freesqldatabase.com'
app.config['MYSQL_USER'] = 'sql12775642'
app.config['MYSQL_PASSWORD'] = 'dWBPNZ32ct'
app.config['MYSQL_DB'] = 'sql12775642'
app.config['MYSQL_PORT'] = 3306
mysql = MySQL(app)

# API keys from environment variables


# With this:
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# 2. Improve the temp directory handling:
TEMP_DIR = os.path.join(os.getenv('RENDER_DISK_PATH', os.getcwd()), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)
os.chmod(TEMP_DIR, 0o777)  # Ensure write permissions

# 3. Add diagnostic endpoints to help troubleshoot:
@app.route('/system-check')
def system_check():
    """Check system dependencies"""
    results = {}
    
    # Check Tesseract
    try:
        results["tesseract"] = {
            "path": TESSERACT_PATH,
            "exists": os.path.exists(TESSERACT_PATH),
            "version": subprocess.check_output([TESSERACT_PATH, '--version'], stderr=subprocess.STDOUT).decode().strip()
        }
    except Exception as e:
        results["tesseract"] = {"error": str(e)}
    
    # Check FFmpeg
    try:
        results["ffmpeg"] = {
            "version": subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT).decode().split('\n')[0]
        }
    except Exception as e:
        results["ffmpeg"] = {"error": str(e)}
    
    # Check temp directory
    results["temp_dir"] = {
        "path": TEMP_DIR,
        "exists": os.path.exists(TEMP_DIR),
        "writable": os.access(TEMP_DIR, os.W_OK),
        "disk_free": shutil.disk_usage(TEMP_DIR).free
    }
    
    # Check API key
    results["groq_api"] = {
        "key_set": bool(GROQ_API_KEY)
    }
    
    return jsonify(results)


# Configure Tesseract path
try:
    TESSERACT_PATH = subprocess.check_output(['which', 'tesseract']).decode().strip()
    logger.info(f"Found Tesseract at: {TESSERACT_PATH}")
except Exception as e:
    logger.warning(f"Could not auto-detect Tesseract: {str(e)}")
    # Try common paths
    possible_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/app/.apt/usr/bin/tesseract'
    ]
    TESSERACT_PATH = next((p for p in possible_paths if os.path.exists(p)), 
                          os.getenv('TESSERACT_PATH', '/usr/bin/tesseract'))
    logger.info(f"Using Tesseract path: {TESSERACT_PATH}")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Create temp directory if it doesn't exist
TEMP_DIR = os.path.join(os.getenv('RENDER_DISK_PATH', os.getcwd()), 'temp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Create directories for templates and static files if they don't exist
TEMPLATE_DIR = os.path.join(os.getcwd(), 'templates')
STATIC_DIR = os.path.join(os.getcwd(), 'static')
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Function to get database connection
def get_db_connection():
    return mysql.connection
@app.route('/features')
def features():
    return render_template("features.html")
@app.route('/doc')
def doc():
    return render_template("doc.html")
@app.route('/faq')
def faq():
    return render_template("faq.html")
@app.route('/')
def home():
    """Render home page"""
    return render_template("entry.html")
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'GET':
        return render_template("login.html")
    
    try:
        logger.info("Login request received")
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400
            
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            logger.warning("Missing email or password")
            return jsonify({"success": False, "message": "Email and password are required"}), 400
        
        if not is_valid_email(email):
            return jsonify({"success": False, "message": "Invalid email format"}), 400
        
        # Using MySQL connection
        cursor = mysql.connection.cursor()
        query = "SELECT id, fstname, lstname, email, password FROM register WHERE email=%s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()
        cursor.close()
        
        if not user:
            logger.warning(f"No user found with email: {email}")
            return jsonify({"success": False, "message": "User not found"}), 404
        
        user_id, firstname, lastname, user_email, hashed_password = user
        
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            session['user_id'] = user_id
            logger.info(f"User {email} logged in successfully")
            return jsonify({"success": True, "message": "Login successful", "user_id": user_id}), 200
        else:
            logger.warning(f"Invalid password attempt for {email}")
            return jsonify({"success": False, "message": "Invalid password"}), 401
        
    except Exception as e:
        logger.error("Login error: %s", str(e))
        return jsonify({"success": False, "message": str(e)}), 500
        
@app.route('/chatbot')
def chatbot():
    return render_template("chatbot.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if request.method == 'GET':
        return render_template("register.html")
    try:
        logger.info("Registration request received")
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400
            
        fstname = data.get("fstname")
        lstname = data.get("lstname")
        email = data.get("email")
        password = data.get("password")
        
        if not all([fstname, lstname, email, password]):
            logger.warning("Missing registration fields")
            return jsonify({"success": False, "message": "All fields are required"}), 400
        
        if not is_valid_email(email):
            return jsonify({"success": False, "message": "Invalid email format"}), 400
            
        if len(password) < 8:
            return jsonify({"success": False, "message": "Password must be at least 8 characters long"}), 400
            
        # Using MySQL connection
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id FROM register WHERE email=%s", (email,))
        if cursor.fetchone():
            logger.warning(f"Email already registered: {email}")
            cursor.close()
            return jsonify({"success": False, "message": "Email already registered"}), 409
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        query = "INSERT INTO register (fstname, lstname, email, password) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (fstname, lstname, email, hashed_password))
        mysql.connection.commit()
        user_id = cursor.lastrowid
        cursor.close()
        logger.info(f"New user registered: {email}")
        
        return jsonify({"success": True, "message": "Registration successful", "user_id": user_id}), 201
        
    except Exception as e:
        logger.error("Registration error: %s", str(e))
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/main', methods=['GET','POST'])
def main():
    return render_template("dark.html")    

@app.route('/imgtxt', methods=['POST'])
def imgtxt():
    """Extract text from an uploaded image with improved timeout and memory handling"""
    temp_files = []  # Track all temporary files for cleanup
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image_file = request.files['image']
        logger.info("Image uploaded: %s", image_file.filename)
        
       
        
        def process_with_memory_limit(image_path):
            """Process image with reduced size to limit memory usage"""
            try:
                # Read image as grayscale directly (uses less memory)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    logger.warning("Could not read image with OpenCV, trying PIL")
                    # Try with PIL if OpenCV fails
                    pil_img = Image.open(image_path).convert('L')
                    img = np.array(pil_img)
                    pil_img.close()
                
                # Get dimensions
                height, width = img.shape
                
                # Set very conservative size limits
                max_dimension = 1200
                max_pixels = 1200 * 1200  # ~1.4 MP
                
                # Check if image exceeds either limit
                if height * width > max_pixels or height > max_dimension or width > max_dimension:
                    # Calculate scale to fit within max pixels
                    scale = min(
                        (max_pixels / (height * width)) ** 0.5,
                        max_dimension / max(height, width)
                    )
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # Resize with NEAREST interpolation (fastest and lowest memory)
                    img = cv2.resize(img, (new_width, new_height), 
                                   interpolation=cv2.INTER_NEAREST)
                    
                    logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                    
                # Apply minimal image enhancements
                img = cv2.GaussianBlur(img, (3, 3), 0)
                
                # Apply simple binary threshold instead of adaptive (uses less memory)
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Save processed image to a new temp file
                processed_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                processed_path = processed_temp.name
                processed_temp.close()
                temp_files.append(processed_path)  # Track for cleanup
                
                cv2.imwrite(processed_path, img)
                return processed_path
                
            except Exception as e:
                logger.error(f"Image preprocessing failed: {str(e)}")
                return image_path  # Return original if processing fails
        
        # Create a temporary file with the same extension as the original
        _, file_extension = os.path.splitext(image_file.filename)
        temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        temp_files.append(temp_file_path)  # Track for cleanup
        
        # Save the uploaded file to the temporary file
        image_file.save(temp_file_path)
        
        # Process image to reduce memory usage
        processed_path = process_with_memory_limit(temp_file_path)
        
        # Define a very strict timeout function with multi-language support
        def extract_text_with_strict_timeout(image_path, timeout=15):
            """Extract text with a strict timeout using alarm signal with multi-language support"""
            import pytesseract
            
            result = None
            timed_out = False
            
            def timeout_handler(signum, frame):
                nonlocal timed_out
                timed_out = True
                raise TimeoutError("OCR timed out")
            
            # First, try to detect the language
            try:
                # Use a quicker method to detect script/language
                pil_img = Image.open(image_path).convert('L')
                sample = pytesseract.image_to_osd(pil_img)
                pil_img.close()
                
                # Try to extract script info
                script_match = re.search(r'Script: ([a-zA-Z]+)', sample)
                script = script_match.group(1).lower() if script_match else 'latin'
                
                logger.info(f"Detected script: {script}")
                
                # Map script to language options
                lang_map = {
                    'latin': 'eng+fra+spa+deu+ita+por',
                    'arabic': 'ara',
                    'cyrillic': 'rus+ukr+bul',
                    'han': 'chi_sim+chi_tra+jpn+kor',
                    'devanagari': 'hin+san',
                    'thai': 'tha',
                    'japanese': 'jpn',
                    'korean': 'kor',
                    'hebrew': 'heb',
                    'greek': 'grc+ell'
                }
                
                # Default to a broad set of languages if script not recognized
                lang_option = lang_map.get(script, 'eng+fra+spa+deu+ita+por+ara+rus+chi_sim+jpn')
                
            except Exception as e:
                logger.warning(f"Language detection failed: {str(e)}, using default languages")
                # If language detection fails, use common languages
                lang_option = 'eng+fra+spa+deu+ita+por'
            
            # Set timeout handler
            try:
                original_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                
                # OCR call with language-specific options
                logger.info(f"Running OCR with languages: {lang_option}")
                result = pytesseract.image_to_string(
                    image_path,
                    config=f'--oem 1 --psm 3 -l {lang_option} --dpi 300'
                )
                
                # Cancel alarm
                signal.alarm(0)
                # Restore original handler
                signal.signal(signal.SIGALRM, original_handler)
                
            except TimeoutError:
                logger.warning("OCR operation timed out")
                return "OCR process timed out. The image may be too complex or unclear."
            except Exception as e:
                logger.error(f"OCR failed: {str(e)}")
                return f"Failed to extract text: {str(e)}"
            
            return result if result else "No text could be extracted from the image."
        
        # Extract text with strict timeout
        start_time = time.time()
        text = extract_text_with_strict_timeout(processed_path)
        logger.info(f"OCR completed in {time.time() - start_time:.2f} seconds")
        
        # Check if text extraction failed or returned empty result
        if not text or text.strip() == "" or "timed out" in text:
            # Fall back to a simpler OCR method if available
            try:
                from PIL import Image
                import pytesseract
                
                logger.info("Trying fallback OCR method")
                pil_img = Image.open(processed_path).convert('L')
                simple_text = pytesseract.image_to_string(pil_img, config='--psm 6')
                pil_img.close()
                
                if simple_text and simple_text.strip() != "":
                    text = simple_text
                else:
                    text = "No text could be extracted from the image."
            except Exception as e:
                logger.error(f"Fallback OCR failed: {str(e)}")
                # Keep original text message
        
        # Use a separate timeout for the LLM call
        try:
            # Process with LLM with proper multi-language handling
            client = Groq(api_key="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")
            
            # Detect possible language family for better instruction
            def detect_language_family(text_sample):
                # Simple script detection based on character ranges
                scripts = {
                    'latin': lambda c: 0x0000 <= ord(c) <= 0x024F,
                    'cyrillic': lambda c: 0x0400 <= ord(c) <= 0x04FF,
                    'arabic': lambda c: 0x0600 <= ord(c) <= 0x06FF,
                    'devanagari': lambda c: 0x0900 <= ord(c) <= 0x097F,
                    'chinese': lambda c: 0x4E00 <= ord(c) <= 0x9FFF,
                    'japanese': lambda c: 0x3040 <= ord(c) <= 0x30FF,
                    'korean': lambda c: 0xAC00 <= ord(c) <= 0xD7AF,
                    'thai': lambda c: 0x0E00 <= ord(c) <= 0x0E7F
                }
                
                # Count characters in each script
                counts = {script: 0 for script in scripts}
                for char in text_sample:
                    for script, check in scripts.items():
                        if check(char):
                            counts[script] += 1
                
                # Return the most common script
                if not counts or max(counts.values()) == 0:
                    return 'unknown'
                return max(counts, key=counts.get)
            
            # Get a sample of the text for language detection (first 100 chars)
            text_sample = text[:100] if len(text) > 100 else text
            lang_family = detect_language_family(text_sample)
            logger.info(f"Detected language family: {lang_family}")
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "system",
                    "content": f"You are a language expert specialized in {lang_family} languages. Create a brief, beautiful summary of the text while preserving its original language and style."
                },
                {
                    "role": "user",
                    "content": f"Create a beautiful summary of this text, maintaining its original language and cultural nuances:\n\n{text}"
                }],
                max_tokens=200  # Limit response size
            )
            summary = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM processing failed: {str(e)}")
            summary = text  # Fallback to the original text if LLM fails
        
        return jsonify({"txt": summary})
    
    except Exception as e:
        logger.error("Error in image to text: %s", str(e))
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
    
    finally:
        # Clean up all temporary files
        for file_path in temp_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Temporary file removed: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove temporary file {file_path}: {str(e)}")

def extract_pdf_in_chunks(pdf_path, chunk_size=4000, by_pages=False):
    """
    Extract text from a PDF in chunks.
    - chunk_size: Number of characters per chunk (if by_pages=False) or pages per chunk (if by_pages=True).
    - by_pages: If True, split by number of pages instead of characters.
    Returns a list of text chunks.
    """
    chunks = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = min(100, len(pdf.pages))  # Limit to 100 pages
            full_text = ""
            
            # Extract text from all pages first
            for page_num in range(total_pages):
                page_text = pdf.pages[page_num].extract_text() or ""
                full_text += page_text + "\n"
            
            if by_pages:
                # Split by number of pages
                for i in range(0, total_pages, chunk_size):
                    start_page = i
                    end_page = min(i + chunk_size, total_pages)
                    chunk_text = ""
                    for j in range(start_page, end_page):
                        page_text = pdf.pages[j].extract_text() or ""
                        chunk_text += page_text + "\n"
                    if chunk_text.strip():
                        chunks.append(chunk_text)
            else:
                # Split by character count
                for i in range(0, len(full_text), chunk_size):
                    chunk = full_text[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                        
        return chunks
    
    except Exception as e:
        logger.error("Error extracting PDF text: %s", str(e))
        return [f"Error extracting text: {str(e)}"]

@app.route('/ytaudio', methods=['POST'])
def youtube_audio_to_text():
    """Extract and transcribe audio from YouTube videos"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "URL is required"}), 400
            
        url = data.get("url", "")
        logger.info("Processing YouTube URL: %s", url)
        
        transcript = process_youtube_audio(url, language="en-US")
        
        if not transcript or not isinstance(transcript, str) or transcript.strip() == "":
            return jsonify({"error": "Failed to generate transcript"}), 400

        # Initialize the Groq client
        client = Groq(api_key="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")

        # Create a chat completion request
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"As a news journalist, summarize this text for a general audience in bullet points highlighting the main ethical points and even give your own opinion on this with proper headings: [{transcript}]"
            }]
        )

        # Get the summarized text
        summary = response.choices[0].message.content
        
        # Return the response as JSON
        return jsonify({
            "transcript": transcript,
            "summary": summary
        })
    except Exception as e:
        logger.error("YouTube processing error: %s", str(e))
        return jsonify({"error": f"Failed to process YouTube video: {str(e)}"}), 500

def process_youtube_audio(url, language="en-US"):
    """Download and transcribe audio from YouTube"""
    temp_audio_file = os.path.join(TEMP_DIR, f"yt_{int(os.urandom(4).hex(), 16)}.wav")
    
    try:
        audio_file = download_audio(url, temp_audio_file)
        if not audio_file:
            logger.error("Failed to download audio from YouTube")
            return None

        transcript = transcribe_in_chunks(audio_file, language=language)
        logger.info("Transcription completed")
        return transcript
        
    except Exception as e:
        logger.error("Error in YouTube audio processing: %s", str(e))
        return None
    finally:
        # Cleanup the audio file
        if os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                logger.info("Audio file deleted: %s", temp_audio_file)
            except Exception as e:
                logger.error("Failed to delete audio file: %s", str(e))

def transcribe_in_chunks(audio_path, chunk_length_ms=90000, language="en-US"):
    """Break audio into chunks and transcribe each chunk"""
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    chunks = make_chunks(audio, chunk_length_ms)
    full_transcript = ""

    logger.info("Splitting audio into %d chunks...", len(chunks))

    for i, chunk in enumerate(chunks):
        logger.info("Transcribing chunk %d/%d...", i+1, len(chunks))
        chunk_path = os.path.join(TEMP_DIR, f"chunk_{i}.wav")
        
        try:
            chunk.export(chunk_path, format="wav")

            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio_data, language=language)
                full_transcript += text + " "
            except sr.UnknownValueError:
                full_transcript += "[Unrecognized] "
                logger.warning("Chunk %d was not recognized", i+1)
            except sr.RequestError as e:
                full_transcript += f"[Request Error] "
                logger.error("Request error in chunk %d: %s", i+1, str(e))
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    return full_transcript.strip()

def download_audio(url, output_path):
    """Download audio from YouTube URL"""
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': os.path.join(TEMP_DIR, 'temp.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        temp_file = os.path.join(TEMP_DIR, "temp.wav")
        if os.path.exists(temp_file):
            os.rename(temp_file, output_path)
            return output_path
        else:
            logger.error("Downloaded audio file not found")
            return None
    except Exception as e:
        logger.error("Error downloading audio: %s", str(e))
        return None

@app.route('/upload-pdf', methods=['POST'])
def pdfimg():
    """Process uploaded PDF files"""
    try:
        # Check if file is in the request
        if 'pdf' not in request.files:
            return jsonify({"error": "No PDF file in request"}), 400
        
        pdf_file = request.files['pdf']
        
        # Check if file was selected
        if pdf_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Secure the filename
        filename = secure_filename(pdf_file.filename)
        temp_path = os.path.join(TEMP_DIR, filename)
        
        # Save the uploaded file temporarily
        pdf_file.save(temp_path)
        logger.info("PDF saved temporarily: %s", temp_path)
        
        try:
            # Process the PDF
            chunk_size = 4000  # Characters per chunk
            text_chunks = extract_pdf_in_chunks(temp_path, chunk_size)
            
            # Check if there was an error during extraction
            if not text_chunks:
                return jsonify({"error": "Failed to extract text, no chunks returned"}), 400
                
            if any(chunk.startswith("Error") for chunk in text_chunks):
                error_chunk = next(chunk for chunk in text_chunks if chunk.startswith("Error"))
                return jsonify({"error": error_chunk}), 400
            
            # Initialize the Groq client
            client = Groq(api_key="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")
            
            # Process all chunks and collect summaries
            all_summaries = []
            
            for i, text in enumerate(text_chunks, 1):
                logger.info("Processing PDF chunk %d/%d", i, len(text_chunks))
                # Create a chat completion request
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{
                        "role": "user",
                        "content": f"Summarize this with main topics and heading with bullet points: [{text}]"
                    }]
                )
                summary = response.choices[0].message.content
                all_summaries.append(summary)
                
            return jsonify({"pdf_summaries": all_summaries})
            
        except Exception as e:
            logger.error("Error processing PDF: %s", str(e))
            return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("PDF temporary file removed")
            except Exception as e:
                logger.error("Failed to remove temp PDF file: %s", str(e))

@app.route('/vidtotxt', methods=['POST'])
def vidtotxt():
    """Convert video to text"""
    temp_video_path = None
    audio_path = None
    
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        # Get the uploaded video file
        video_file = request.files.get("video")
        
        # Create unique filenames
        file_id = int(os.urandom(4).hex(), 16)
        temp_video_path = os.path.join(TEMP_DIR, f"temp_video_{file_id}.mp4")
        audio_path = os.path.join(TEMP_DIR, f"output_audio_{file_id}.wav")
        
        # Save the uploaded file temporarily
        video_file.save(temp_video_path)
        logger.info("Video saved temporarily: %s", temp_video_path)
        
        # Extract audio from video
        try:
            video = moviepy.VideoFileClip(temp_video_path)
            video.audio.write_audiofile(audio_path, logger=None)  # Suppress moviepy logs
            video.close()  # Close the video file
        except Exception as e:
            logger.error("Failed to extract audio from video: %s", str(e))
            return jsonify({"error": f"Failed to extract audio: {str(e)}"}), 500
        
        # Transcribe the audio in chunks for better handling of large files
        try:
            transcript = transcribe_in_chunks(audio_path)
            logger.info("Video transcription completed")
            return jsonify({"msg": transcript})
        except Exception as e:
            logger.error("Failed to transcribe audio: %s", str(e))
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        
    except Exception as e:
        logger.error("Video to text error: %s", str(e))
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary files
        cleanup_files([temp_video_path, audio_path])

def cleanup_files(file_paths):
    """Remove temporary files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Removed temp file: %s", file_path)
            except Exception as e:
                logger.error("Error removing file %s: %s", file_path, str(e))

@app.route('/audtotxt', methods=['POST'])
def audtotxt():
    """Convert audio to text"""
    audio_path = None
    wav_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        
        # Create unique filenames
        file_id = int(os.urandom(4).hex(), 16)
        audio_path = os.path.join(TEMP_DIR, f"uploaded_audio_{file_id}.mp3")
        wav_path = os.path.join(TEMP_DIR, f"converted_{file_id}.wav")
        
        # Save the uploaded file
        audio_file.save(audio_path)
        logger.info("Audio saved temporarily: %s", audio_path)

        # Convert to WAV format
        try:
            sound = AudioSegment.from_file(audio_path)  # More general than from_mp3
            sound.export(wav_path, format="wav")
        except Exception as e:
            logger.error("Failed to convert audio format: %s", str(e))
            return jsonify({"error": f"Failed to convert audio: {str(e)}"}), 500

        # Transcribe the audio in chunks for better handling
        try:
            transcript = transcribe_in_chunks(wav_path)
            logger.info("Audio transcription completed")
            return jsonify({"message": transcript})
        except Exception as e:
            logger.error("Failed to transcribe audio: %s", str(e))
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    except Exception as e:
        logger.error("Audio to text error: %s", str(e))
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up temporary files
        cleanup_files([audio_path, wav_path])

# Helper function to validate email format
def is_valid_email(email):
    """Check if email has a valid format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

         

@app.route('/logout', methods=['GET','POST'])
def logout():
    """Handle user logout"""
    try:
        # Clear the session
        session.pop('user_id', None)
        logger.info("User logged out")
        return jsonify({"success": True, "message": "Logout successful"}), 200
    except Exception as e:
        logger.error("Logout error: %s", str(e))
        return jsonify({"success": False, "message": str(e)}), 500



@app.route('/txtsumz', methods=['POST'])
def txtsumz():
    """Summarize text input"""
    try:
        logger.info("Text summarization request received")
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "No text message provided"}), 400
            
        message = data.get("message", "")
        
        if not message.strip():
            return jsonify({"error": "Empty text cannot be summarized"}), 400
        
        client = Groq(api_key="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")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"summarize this in a very beautiful in the language the input is provided:{message}"
            }]
        )

        summary = response.choices[0].message.content
        logger.info("Text summarization completed")
        return jsonify({"message": summary})
        
    except Exception as e:
        logger.error("Text summarization error: %s", str(e))
        return jsonify({"error": f"Failed to summarize text: {str(e)}"}), 500


# Database check endpoint for debugging
@app.route('/db-check')
def db_check():
    """Debug endpoint to check database connection"""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        cursor.close()
        return jsonify({"connected": True, "version": version[0]})
    except Exception as e:
        return jsonify({"connected": False, "error": str(e)}), 500

# Create a simple health check endpoint for Render
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200

# Create database tables if they don't exist
def init_db():
    try:
        cursor = mysql.connection.cursor()
        
        # Create register table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS register (
            id INT AUTO_INCREMENT PRIMARY KEY,
            fstname VARCHAR(100) NOT NULL,
            lstname VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        mysql.connection.commit()
        cursor.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
@app.route('/screenpipe')
def sumz():
    return render_template("index.html")

# Initialize the database on startup
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # Initialize database when running in production (only if in a context where mysql is available)
    with app.app_context():
        init_db()
