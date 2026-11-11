"""Frozen matched fine-tune ensembles with previous-stage controls."""
import json
from pathlib import Path
import numpy as np
import torch
from cranepilot.broad_strategies import make_model
from scripts.global_research_candidate import GlobalResearchCandidate, CONTROLS
from scripts.run_broad_search import inputs, predict


class FinetuneCandidate(GlobalResearchCandidate):
    def __init__(self, config, fold, directory=None):
        super().__init__(config,fold,Path(config['parent_run'])/fold)
        previous=json.loads((Path(config['parent_run'])/'selection.json').read_text())
        self.previous_name=previous['folds'][fold]['model']
        self.ensembles={}
        if directory is not None:
            for name in config['variants']:
                models=[]
                for seed in config['seeds']:
                    model=make_model({'symmetry_mix':.5})
                    model.load_state_dict(torch.load(Path(directory)/f'{name}_seed{seed}.pt',weights_only=True))
                    models.append(model.eval())
                self.ensembles[name]=models

    def forecast(self,raw,prescribed_acceleration=None):
        prior,acc=super().forecast(raw,prescribed_acceleration)
        predictions={name:prior[name] for name in CONTROLS}
        predictions['previous_local']=prior[self.previous_name]
        batch=inputs(raw);batch['acceleration']=torch.as_tensor(acc,dtype=torch.float32)
        seeds={name:np.stack([predict(model,batch) for model in models]) for name,models in self.ensembles.items()}
        predictions.update({name:value.mean(0) for name,value in seeds.items()})
        if not all(y.shape==(len(raw['omega']),30) and np.isfinite(y).all() for y in predictions.values()):
            raise ValueError('Invalid frozen fine-tune forecast')
        return predictions,seeds,acc
