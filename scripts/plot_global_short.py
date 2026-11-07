"""Reproduce horizon-specific gains and regressions of the frozen short-fold choice."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.run_global_research import verify


def main():
    run=Path('artifacts/runs/global_research_20261107');out=Path('results/global_research_20261107')
    prov=json.loads((run/'provenance.json').read_text())
    if prov['status']!='complete':raise ValueError('Completed run required')
    verify(prov['result_hashes'],run)
    selected=json.loads((run/'selection.json').read_text())['folds']['short_transfer']['model']
    data=pd.read_csv(run/'evaluate_metrics.csv')
    data=data[(data.fold=='short_transfer')&(data.group=='evaluation')&data.extrapolation&data.model.isin(['incumbent',selected])]
    detail=data.groupby(['regime','horizon_s','model']).mae_deg.mean().unstack()
    detail['change_percent']=100*(detail[selected]/detail.incumbent-1)
    detail.to_csv(out/'short_transfer_horizons_detail.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,regime,title in zip(axes,['all','active'],['Все окна','Тележка движется']):
        local=detail.loc[regime]
        ax.plot(local.index,local.incumbent,'o-',color='#6b7280',label='Сохранённая модель',lw=2)
        ax.plot(local.index,local[selected],'o-',color='#127f79',label='Новая LPV',lw=2)
        ax.set(xlabel='Горизонт прогноза, с',ylabel='MAE угла, °',title=title,xlim=(.05,1.55))
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
    fig.suptitle('Перенос на короткие тросы: результат зависит от горизонта',fontsize=14)
    fig.savefig(out/'short_transfer_horizons_detail.png',dpi=180)
    fig.savefig(out/'short_transfer_horizons_detail.svg')
    plt.close(fig)
    print(detail.to_string())


if __name__=='__main__':main()
