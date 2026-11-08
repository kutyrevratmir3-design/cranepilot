"""Supplement the preserved original plot with every reference, including dated modal."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256

OUT=Path('results/latency_observer_20261108')
def main():
    dest=OUT/'latency_all_baselines.png'
    if dest.exists():raise FileExistsError('Preserve previous figure')
    data=OUT/'age_noise_comparison.csv';frame=pd.read_csv(data);fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained');methods=[('unaware','Age ignored','#9caaa9','-'),('timestamp_hold','Timestamp shift + hold','#648b6d','-'),('selected','Frozen selected adapter','#bb8247','-'),('dated_modal','Dated free modal','#9a6f48','--'),('matched_arx','Global ARX reference','#3f7183','-'),('conditional_arx','Late age-conditioned ARX','#735c83','-')]
    for j,noise in enumerate([0,1]):
        table=frame[frame.noise_quantum==noise].pivot(index='age_ms',columns='method',values='mae_deg').sort_index()
        for key,label,color,style in methods:axs[j].plot(table.index,table[key],label=label,color=color,linestyle=style,marker='o',markersize=4)
        axs[j].set_xlabel('Known angle measurement age (ms)');axs[j].set_ylabel('MAE (degrees), equal old scenarios/axes/horizons');axs[j].set_title('Clean history' if noise==0 else 'Synthetic angle-history jitter ±q');axs[j].grid(alpha=.2)
    axs[0].legend(frameon=False);fig.suptitle('All fixed baselines; dated modal slightly wins at150ms. Retrospective qualification.');fig.savefig(dest,dpi=170);plt.close(fig);(OUT/'all_baselines_plot.json').write_text(json.dumps(dict(source_sha256=sha256(Path(__file__)),data_sha256=sha256(data),image_sha256=sha256(dest),original_plot_preserved=True,methods=[m[0] for m in methods],independent_confirmation=False),indent=2)+'\n')
if __name__=='__main__':main()
