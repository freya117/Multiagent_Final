import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook(notebook_path):
    print(f"Running {notebook_path}...")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        print(f"✓ {notebook_path} passed!")
        return True
    except Exception as e:
        print(f"✗ {notebook_path} failed!")
        print(e)
        return False

if __name__ == "__main__":
    notebooks = [
        "testing.ipynb",
        "leduc_experiments.ipynb",
        "general_sum_analysis.ipynb"
    ]
    
    for nb in notebooks:
        run_notebook(os.path.abspath(nb))
