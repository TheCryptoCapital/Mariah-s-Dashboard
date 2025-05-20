setup.py  
#!/usr/bin/env python3
import subprocess
import sys
import os

def install_dependencies():
    print("Installing core dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("\nWould you like to install optional features? (y/n)")
    choice = input().lower()
    if choice.startswith('y'):
        print("Installing optional dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-optional.txt"])
    
    print("\nSetting up environment...")
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Crypto Capital Dashboard Environment Variables\n")
            f.write("OPENAI_API_KEY=\n")
            f.write("API_KEY=\n")
            f.write("API_SECRET=\n")
            f.write("EMAIL_USER=\n")
            f.write("EMAIL_PASSWORD=\n")
            f.write("EMAIL_HOST=\n")
            f.write("EMAIL_PORT=\n")
        print("Created .env file. Please edit it to add your API keys.")
    
    print("\nSetup complete! Run the dashboard with: streamlit run dashboard.py")

if __name__ == "__main__":
    install_dependencies()
    