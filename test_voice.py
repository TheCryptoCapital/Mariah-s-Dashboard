import subprocess

def speak_macos(text):
    try:
        subprocess.call(['say', text])
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test it
speak_macos("Hello, I am Mariah. Testing voice on Mac OS.")