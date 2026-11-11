"""Display the conservative reference, including the scenario21 failure."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=Path('results/modal_reference_challenge_20261107');target=out/'conservative_comparison.png'
    if target.exists():raise FileExistsError('Retain prior figure')
    records=pd.read_csv(out/'record_summary.csv');m=records.pivot(index='scenario',columns='method',values='mae_deg');x=np.arange(len(m));fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    ax.bar(x-.18,m.challenged_arx,.36,label='Strengthened simple ARX',color='#3f7183');ax.bar(x+.18,m.selected_modal,.36,label='Frozen two-mode predictor',color='#bb8247');ax.set_xticks(x,[str(i) for i in m.index]);ax.set_xlabel('Public scenario (all now retrospective)');ax.set_ylabel('MAE (degrees), equal axes / horizons');ax.set_title('Residual-sway forecast: same 4-second history');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.savefig(target,dpi=170);plt.close(fig)
if __name__=='__main__':main()
