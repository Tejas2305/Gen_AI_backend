import os
import sys
from app import app

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    app.run()
