![github-submission-banner](https://github.com/user-attachments/assets/a1493b84-e4e2-456e-a791-ce35ee2bcf2f)

# 🚀Welcome to Summarizer+
Summarizer+ is a website to save your time by summarizing texts,audio,video,pdfs,YouTube links and can also capture your devices screen to extract text and summarize contents on the screen

## 📌 Problem Statement
- Weave AI magic with groq
- Build the future of AI computer control with screenpipe's terminator

## 🎯 Objective
our objective in this hackathon is to build a reliable AI powered Summarizer which can extract text from media sources like audio,video,images,pdfs and even youtube videos!
our goal is to save time of users by summarizing any content for quick understanding to save the users time in this busy world

## 🧠 Team & Approach
### Team Name:
`Team Ampersand &&`

### Team Members:  
- S Rohan Kumar (GitHub: https://github.com/S-Rohan-Kumar / linkedin: https://www.linkedin.com/in/s-rohan-kumar-a1b59132b/ / Team Leader and Main developer)
  
- Srujan N (GitHub: https://github.com/planeloverpilot / linkedin: https://www.linkedin.com/in/srujan-n-95603332a/ / Webpage Designing and feature handling)
  
- Sudhir A (GitHub:  / linkedin: / Screenpipe integration developer)
  
- Preran S (GitHub: https://github.com/PRERAN01 / linkedin: https://www.linkedin.com/in/preran-s-131077345/ / Database and Website Connection Manager )

### Your Approach:  
- Why you chose this problem: we chose this problem as it would help us understand the need of AI in the modern day and its potential as a time saver and to provide more information in less content
  
- Key challenges you addressed: integrating screenpipe
  
- Any pivots, brainstorms, or breakthroughs during hacking: made an extra chatbot feature for any immediate requests from AI

  ---

## 🛠️ Tech Stack

### Core Technologies Used:
- Frontend: HTML and CSS and JavaScript(js)
- Backend: python
- Database: mySQL
- APIs: groq, deepseek and screenpipe
- Hosting: Render Hosting

### Sponsor Technologies Used (if any):
- [✅] **Groq:** _we used groq to summarize text_  

- [✅] **Screenpipe:** _Screen-based analytics or workflows_  
---

## ✨ Key Features

Highlight the most important features of your project:

- ✅ Feature 1: Text extraction from multiple medias like images,videos,audio,pdfs and audio extraction from YouTube links  
- ✅ Feature 2: Live screen text extraction using screenpipe terminator
- ✅ Feature 3: Mini AI chatbot option for extra assistance or for general help using Groq API
- ✅ Feature 4: Text summarizer using groq API
- ✅ Feature 5: Multilingual AI response (kannada, hindi, telegu and tamil)

  ## 📽️ Demo & Deliverables

- **Demo Video Link:** [https://youtu.be/vJIQnhKvQr0]
- **Deployment Link:** [https://thesuumarizerplus.onrender.com]
- (NOTE: due to limited web deployment options, YouTube text summarization and Screenpipe are broken as the server cannot fetch youtube video links and doesnt allow display capture for screenpipe, the features work if ran on localhost as shown in the demo video)
- **Pitch Deck / PPT Link:** [projectsumz^M.pdf](https://github.com/user-attachments/files/19799532/projectsumz.M.pdf)  

---

## ✅ Tasks & Bonus Checklist

- [✅] **All members of the team completed the mandatory task - Followed at least 2 of our social channels and filled the form** (Details in Participant Manual)  
- [ ] **All members of the team completed Bonus Task 1 - Sharing of Badges and filled the form (2 points)**  (Details in Participant Manual)
- [ ] **All members of the team completed Bonus Task 2 - Signing up for Sprint.dev and filled the form (3 points)**  (Details in Participant Manual)

## 🧪 How to Run the Project

---

### Requirements:
- Python and Docker
- API Keys: Groq (api_key="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") , deepseek(api_key="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"), screenpipe
- .env file setup (if needed)
  reccomended
  
✅ Prerequisites

- Ensure you have the following installed:

- Python 3.9+

- pip

- MySQL Server

- Tesseract OCR

- ffmpeg

- Screenpipe.exe (seems to be a Windows screen capture utility)

- Optional: Node.js if your templates use any JS-based tools

- pip install flask flask-cors flask-mysqldb pytesseract pillow moviepy SpeechRecognition pydub yt_dlp bcrypt python-dotenv requests pdfplumber

⚙️ 4. Install External Software

  - Tesseract OCR
  
  - Download and install from: https://github.com/tesseract-ocr/tesseract
  
  - Note the install path and update your .env file like:
  
  - ini
  
  - Copy
  
  - Edit
  
  - TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
  
  - FFmpeg (for moviepy/audio)
  
  - Download: https://ffmpeg.org/download.html
  
  - Add it to your system PATH.
  
  - MySQL
  
  - Create a database named login.
  
  - Create a table register as expected in app1.py.
  
  ✅Example schema:
  
  - sql
  - Copy
  - Edit
    
    CREATE TABLE register (

        id INT AUTO_INCREMENT PRIMARY KEY,

        fstname VARCHAR(255),

        lstname VARCHAR(255),

        email VARCHAR(255) UNIQUE,

        password TEXT
    
    );
    
  ✅Screenpipe
  
  - Ensure Screenpipe.exe is available.
  
  - The app automatically tries to locate and run it via screenpipe_launcher.py.
  
🔑 5. Set Up .env File
  
  ✅Create a .env file in the root folder:

  - env
  - Copy
  - Edit
  - SECRET_KEY=your_secret_key
  - MYSQL_HOST=localhost
  - MYSQL_USER=root
  - MYSQL_PASSWORD=your_mysql_password
  - MYSQL_DB=login
  - TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
  - GROQ_API_KEY=your_groq_api_key
  - DEEPSEEK_API_KEY=your_deepseek_api_key


🚀 6. Run the App
  
  ✅Make sure you run app1.py:
  
  - bash
  - Copy
  - Edit
  - python app1.py
  - Visit: http://localhost:8000




### Local Setup:
```bash
my_project/
├── app1.py
├── main.py
├── langi.py
├── screenpipe_launcher.py
├── templates/      <- Make sure to have these for rendering HTML
├── static/         <- (for CSS/JS/images)
-----------
cd my_project
python -m venv venv
venv\Scripts\activate      # On Windows
# OR
source venv/bin/activate   # On Mac/Linux
------
pip install flask flask-cors flask-mysqldb pytesseract pillow moviepy SpeechRecognition pydub yt_dlp bcrypt python-dotenv requests pdfplumber


```

## 🧬 Future Scope

List improvements, extensions, or follow-up features:

- 📈 More integrations and faster response time  
- 🛡️ Security enhancements  
- 🌐 Localization / broader accessibility

---

## 📎 Resources / Credits

- APIs or datasets used: groq, screenpipe
- Open source libraries or tools referenced: flask, pytesseract, pdfplumber, langi, moviepy, subprocess, yt_dlp
- Acknowledgements  

---

## 🏁 Final Words

We had a fun time making this code from scratch, it was a huge learning experience for a team of 2nd semester students who have never done fullstack web development before, we had days where the code was running smooth, some days where the code broke, but overall it was satisfying and is still looking forward to upgrade and enhance the project and make more in the future

---
