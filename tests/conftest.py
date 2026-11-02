import os
import sys

# Make the project root importable and keep tests away from real keys and real data.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MODEL", "fake:model")
