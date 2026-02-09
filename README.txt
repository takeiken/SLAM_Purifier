🌊 SLAM Purifier Pro v6.1
Automated Marine Video Analysis & AI Description Engine
SLAM Purifier Pro is a high-performance tool designed for marine researchers and ROV pilots. It automatically filters raw underwater footage to extract only the sharpest, most stable segments, optionally using the Moondream2 AI to describe the marine life and scenery.

💻 Installation Guide
1. Prerequisites (All Systems)
Python 3.10 or 3.11 is recommended. (Note: 3.12 is supported but may require manual pip updates for certain AI libraries).
FFmpeg: Required for video processing.

2. Windows Setup
Install FFmpeg: Download from gyan.dev, extract it, and add the /bin folder to your System PATH.
GPU Acceleration (Optional): If you have an NVIDIA GPU, install the CUDA Toolkit.
Install Dependencies: Open PowerShell and run:
pip install customtkinter opencv-python pillow torch torchvision torchaudio transformers accelerate moviepy

3. Mac Setup
Install FFmpeg: Use Homebrew: brew install ffmpeg.
Install Dependencies: Open Terminal and run:
pip install customtkinter opencv-python pillow torch torchvision torchaudio transformers accelerate moviepy
Note: For M1/M2/M3 Macs, Torch will automatically utilize the MPS (Metal Performance Shaders) for AI acceleration.

🚀 How to Use
Step 1: Launch the App
Run the script using:
python slam_purifier_pro.py

Step 2: Configure the Sidebar
Select Folder: Choose the directory containing your .mp4 or .mov files.
Batch Name: Enter a name (e.g., Coral_Survey_Day1). This creates a subfolder for your exports.
AI Engine Toggle: (Default OFF) ON: Automatically downloads/loads the 3.7GB Moondream model to describe your clips/OFF: Rapid processing mode. Skip descriptions to save RAM/Time.

Step 3: Set Thresholds
Adjust the dual-sync sliders or type values into the boxes:
Sharpness (Laplacian): Higher values = stricter focus requirements. (Typical: 60-90).
Stability (Optical Flow): Lower values = stricter movement requirements. (Typical: 3.0-5.0).

Step 4: Process
Click START PROCESS.
The Visual Scope will show real-time ✅/❌ status for every frame.
The Progress Bar tracks the overall batch completion.

📁 Output Structure
The app creates a new folder inside your source directory named [BatchName]_[Timestamp]:
CLEAN_...mp4: The high-quality extracted clips.
Summary_[BatchName]_[Timestamp].csv: A spreadsheet containing:
Timestamps of extracted clips.
AI Descriptions (if enabled).
Processing metrics (Frames/sec, Usability %).

🛠 Troubleshooting
"AI Model Failed to Load": Ensure you have at least 8GB of RAM (16GB recommended) and an active internet connection for the first run to download model weights.
Windows Symlink Warning: v6.1 automatically suppresses this, but running the app as Administrator once can resolve persistent permission issues with HuggingFace models.
Slow Processing: Disable the "AI Engine" toggle to run at 5x speed for basic clip extraction.