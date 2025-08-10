import os
import subprocess
import sys

def test_training_script_runs():
    # run a quick training
    cmd = [sys.executable, "-m", "src.models.train", "--params", "params/iris.yaml"]
    assert subprocess.call(cmd) == 0
    assert os.path.exists("models/registry/model.joblib")
    assert os.path.exists("models/registry/model_meta.json")
