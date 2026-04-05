import sys
import os

# Add project root and src/ to sys.path so all test imports resolve cleanly
# regardless of the working directory pytest is invoked from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
