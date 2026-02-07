import subprocess
import sys

subprocess.run([sys.executable, "sarr_prop_analysis.py"], check=True)

python -m pip install -r requirements.txt
python run_analysis.py