import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001) 