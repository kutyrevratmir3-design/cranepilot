from pathlib import Path
import json
from cranepilot.data import fixed_folds

config=json.loads(Path("configs/extrapolation_pilot.json").read_text())
config["description"]="Retrospective all-family neural development benchmark; same fixed hyperparameters as first pilot."
config["folds"]=fixed_folds()
Path("configs/neural_loso.json").write_text(json.dumps(config,indent=2))
print(f"Saved {len(config['folds'])} neural folds")

