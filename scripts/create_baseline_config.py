from pathlib import Path
import json
from cranepilot.data import fixed_folds

config = {"description":"Retrospective all-family physical baselines; causal corrected indexing.",
          "horizons_steps":[10,50,100],"histories":[3,41],"train_stride":4,"eval_stride":1,
          "seeds":[0,1,2],"epochs":80,"projection_weight":.03,
          "trees":False,"gp":False,"neural":False,"folds":fixed_folds()}
Path("configs/loso_baselines.json").write_text(json.dumps(config,indent=2))
print(f"Saved {len(config['folds'])} scenario/axis folds")

