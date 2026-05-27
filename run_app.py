import subprocess
import sys
import os

os.chdir(r"C:\Users\kun\Desktop\fruit_ripeness_std")

subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--browser.gatherUsageStats", "false",
    "--server.port", "8501",
], check=True)