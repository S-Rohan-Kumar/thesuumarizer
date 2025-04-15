#!/usr/bin/env python3
"""
Launcher script for Screen Content Extractor with DeepSeek AI
This script starts both the Flask API server and opens the frontend in a browser.
"""

import subprocess
import threading
import time
import webbrowser
import os
import sys
import platform
import logging
from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Add CORS support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('launcher.log')
    ]
)
logger = logging.getLogger(__name__)

def is_port_in_use(port):
    """Check if a port is in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def create_frontend_directory():
    """Create the frontend directory if it doesn't exist and copy the index.html file"""
    try:
        if not os.path.exists('frontend'):
            os.makedirs('frontend')
        
        # Check if the index.html file exists in the frontend directory
        if not os.path.exists(os.path.join('frontend', 'index.html')):
            # Copy from the current directory if it exists there
            if os.path.exists('index.html'):
                import shutil
                shutil.copy('index.html', os.path.join('frontend', 'index.html'))
                logger.info("Copied index.html to frontend directory")
            else:
                logger.warning("index.html not found. Frontend may not work properly.")
    except Exception as e:
        logger.error(f"Error creating frontend directory: {e}")

def start_flask_server():
    """Start the Flask API server"""
    try:
        logger.info("Starting Flask API server...")
        
        if platform.system() == "Windows":
            flask_process = subprocess.Popen(["python", "app.py"], 
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            flask_process = subprocess.Popen(["python3", "app.py"], 
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
                                          
        return flask_process
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")
        return None

def check_required_files():
    """Check if all required files exist"""
    required_files = ['app.py', 'main.py', 'screenpipe_launcher.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        print(f"Error: Missing required files: {', '.join(missing_files)}")
        print("Please make sure all required files are in the current directory.")
        return False
    
    return True

def main():
    """Main function to start the application"""
    print("=" * 60)
    print("  Screen Content Extractor with DeepSeek AI - Launcher")
    print("=" * 60)
    
    # Check if required files exist
    if not check_required_files():
        return
    
    # Create frontend directory if needed
    create_frontend_directory()
    
    # Check if port 7000 is already in use
    if is_port_in_use(7000):
        print("Port 7000 is already in use. The Flask API server might already be running.")
        open_browser = input("Do you want to open the frontend in a browser? (y/n): ").strip().lower()
        if open_browser == 'y':
            webbrowser.open("http://localhost:7000")
        return
    
    # Start Flask server
    flask_process = start_flask_server()
    if not flask_process:
        print("Failed to start Flask API server. Check the logs for details.")
        return
    
    print("Starting Flask API server on port 7000...")
    
    # Wait for Flask to start
    max_attempts = 10
    for attempt in range(max_attempts):
        if is_port_in_use(7000):
            print("Flask API server is running!")
            break
        if attempt < max_attempts - 1:
            print(f"Waiting for API server to start ({attempt+1}/{max_attempts})...")
            time.sleep(1)
    else:
        print("Failed to detect Flask API server after multiple attempts.")
        print("The server might be starting slowly or encountering errors.")
        print("Check the logs for details.")
    
    # Open browser
    print("Opening frontend in browser...")
    webbrowser.open("http://localhost:7000")
    
    print("\nApplication is now running!")
    print("- API server: http://localhost:7000/api")
    print("- Frontend: http://localhost:7000")
    print("\nPress Ctrl+C to stop the application")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the application...")
        logger.info("Application shutdown initiated by user")
        
        # Terminate the Flask server process
        if flask_process:
            logger.info("Terminating Flask server...")
            try:
                if platform.system() == "Windows":
                    # On Windows, use taskkill to force terminate the process and its children
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(flask_process.pid)])
                else:
                    # On Unix-like systems, terminate the process group
                    flask_process.terminate()
                    flask_process.wait(timeout=5)  # Wait up to 5 seconds for process to terminate
            except subprocess.TimeoutExpired:
                logger.warning("Flask server did not terminate gracefully, forcing...")
                flask_process.kill()
            except Exception as e:
                logger.error(f"Error terminating Flask server: {e}")
        
        print("Application has been shut down.")
        logger.info("Application shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    main()