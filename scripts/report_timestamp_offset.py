"""Describe verified frozen age offsets; no fit, selection or controller claim."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cranepilot.data import sha256
from scripts.report_angular_control import table
from scripts.run_external_benchmark import verify_hashes, write

RUN = Path('results/timestamp_offset_20261109')
AUDIT = Path('results/timestamp_offset_audit_20261109/verification.json')
OUT = Path('results/timestamp_offset_report_20261109')
DOC = Path('docs/TIMESTAMP_OFFSET_2026_11_09_RU.md')
METHODS = ('selected', 'dense_matched_arx')


def envelope():
    """Oracle-centered timing range; descriptive coverage, not an interval model."""
    rows = []
    for file in sorted((RUN / 'batches').glob('*.npz')):
        case, sid, _ = file.stem.rsplit('_', 2)
        with np.load(file, allow_pickle=False) as z:
            target = z['target_rad']
            for method in METHODS:
                lo, hi = z[method].min(axis=0), z[method].max(axis=0)
                gap = np.maximum(np.maximum(lo-target, target-hi), 0)
                for i, horizon in enumerate(z['horizons_raw']):
                    rows.append(dict(case=case, scenario=int(sid[1:]), true_age_raw=int(z['true_age_raw']),
                        method=method, horizon_raw=int(horizon), offset_count=len(z['offsets_raw']),
                        mean_width_deg=float(np.rad2deg(hi-lo)[:, i].mean()),
                        target_inside_fraction=float((gap[:, i] <= 1e-12).mean()),
                        mean_outside_distance_deg=float(np.rad2deg(gap)[:, i].mean())))
    axis = pd.DataFrame(rows)
    axis.to_csv(OUT/'envelope_axis_horizon.csv', index=False)
    keys = ['true_age_raw', 'method', 'horizon_raw']
    fields = ['mean_width_deg', 'target_inside_fraction', 'mean_outside_distance_deg']
    records = axis.groupby(keys + ['scenario'], as_index=False)[fields].mean()
    mean = records.groupby(keys, as_index=False)[fields].mean()
    mean.to_csv(OUT/'envelope_mean_by_horizon.csv', index=False)
    return mean.groupby(['true_age_raw', 'method'], as_index=False)[fields].mean()


def main():
    p = json.loads((RUN/'provenance.json').read_text())
    v = json.loads(AUDIT.read_text())
    assert p['status'] == 'complete' and v['passed'] and not p['independent_confirmation']
    verify_hashes(p['result_hashes'], RUN)
    if DOC.exists():
        raise FileExistsError('Retain earlier report')
    OUT.mkdir()
    a = pd.read_csv(RUN/'mean_by_true_age_offset.csv')
    routes = pd.read_csv(RUN/'reference_routes.csv')
    dif = a.pivot(index=['true_age_raw','offset_raw'],columns='method',values='mae_deg').reset_index()
    dif['selected_minus_dense_deg'] = dif.selected - dif.dense_matched_arx
    dif.to_csv(OUT/'paired_cell_means.csv', index=False)
    q = pd.read_csv(RUN/'common_interior_offset_means.csv')
    common = q.pivot(index='offset_raw',columns='method',values='mae_deg').reset_index()
    common['offset_ms'] = common.pop('offset_raw')*10
    inc = q.pivot(index='offset_raw',columns='method',values='paired_delta_from_zero_deg').reset_index()
    common['crane_delta_deg'] = inc.selected
    common['arx_delta_deg'] = inc.dense_matched_arx
    common = common[['offset_ms','selected','dense_matched_arx','crane_delta_deg','arx_delta_deg']]
    losses = dif[dif.selected_minus_dense_deg > 0].copy()
    losses.to_csv(OUT/'aggregate_losses.csv',index=False)
    by_horizon = pd.read_csv(RUN/'record_horizon_metrics.csv')
    short = by_horizon[(by_horizon.true_age_raw==25) & (by_horizon.horizon_raw==10)].pivot(
        index=['offset_raw','scenario'],columns='method',values='mae_deg').reset_index()
    short['primary_losses'] = short.selected > short.dense_matched_arx
    short = short.groupby('offset_raw',as_index=False).agg(
        selected=('selected','mean'),dense_matched_arx=('dense_matched_arx','mean'),
        primary_losses=('primary_losses','sum'))
    short['offset_ms'] = short.pop('offset_raw') * 10
    short = short[['offset_ms','selected','dense_matched_arx','primary_losses']]
    short.to_csv(OUT/'short_horizon_at_250ms.csv',index=False)
    stress = a[a.offset_raw!=0].copy()
    worst = stress.loc[stress.groupby('method').paired_delta_from_zero_deg.idxmax()].copy()
    worst['true_age_ms'] = worst.true_age_raw*10
    worst['offset_ms'] = worst.offset_raw*10
    env = envelope()
    env.to_csv(OUT/'envelope_mean_by_age.csv', index=False)
    env_common = env[env.true_age_raw.isin(p['common_true_ages_raw'])].groupby('method',as_index=False).mean(numeric_only=True)
    display_env = env_common[['method','mean_width_deg','target_inside_fraction','mean_outside_distance_deg']]

    ages=sorted(a.true_age_raw.unique())
    offsets=sorted(a.offset_raw.unique())
    scale=max(.01,float(a.paired_delta_from_zero_deg.abs().max()))
    fig, axes=plt.subplots(1,2,figsize=(11,5),layout='constrained',sharey=True)
    for ax,key,title in zip(axes,METHODS,['Frozen CranePilot','Frozen dense ARX']):
        m=a[a.method==key].pivot(index='true_age_raw',columns='offset_raw',values='paired_delta_from_zero_deg').reindex(index=ages,columns=offsets).to_numpy()
        cmap=plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#e6e6e6')
        im=ax.imshow(m,cmap=cmap,vmin=-scale,vmax=scale,aspect='auto')
        for i,j in np.ndindex(m.shape):
            if np.isfinite(m[i,j]):
                ax.text(j,i,f'{m[i,j]:+.3f}',ha='center',va='center',color='white' if abs(m[i,j])>.55*scale else '#151515',fontsize=9)
        ax.set(xticks=np.arange(len(offsets)),xticklabels=[str(d*10) for d in offsets],
               yticks=np.arange(len(ages)),yticklabels=[str(d*10) for d in ages],
               title=title,xlabel='Declared minus true age (ms)')
    axes[0].set_ylabel('True angle age (ms)')
    fig.colorbar(im,ax=axes,label='Paired MAE change from correct age (degrees)',shrink=.82)
    fig.suptitle('Wrong delay labels on previously opened records')
    fig.savefig(OUT/'timestamp_offset_heatmap.png',dpi=175)
    plt.close(fig)
    nonzero_routes = routes[routes.offset_raw!=0]
    changed = int(nonzero_routes.changed_weight_key.sum())
    common_ages = ', '.join(str(n*10) for n in p['common_true_ages_raw'])
    DOC.write_text(f'''# Что меняется, если ошибиться в задержке датчика

9 ноября 2026. Зафиксированные CranePilot и плотный ARX получили одинаковые
истории и цели из семи ранее открытых записей. Меняли только переданную модели
задержку угла: на ±10 или ±20 мс. Данные датчика, длина, команды и координаты
опоры не сдвигались. Положительная ошибка означает, что модели сказали, будто
угол старше, чем он есть. Никакие веса и правила выбора не менялись.

## Изменение ошибки

Сравнение на одинаковых истинных задержках **{common_ages} мс**, для которых
доступны все пять смещений. Равные сценарии, оси внутри записи, горизонты
и истинные задержки. MAE и изменение относительно верной метки в градусах:

{table(common)}

На полной допустимой сетке {len(dif)} пар «истинная задержка, смещение»
CranePilot уступает ARX в {len(losses)} средних ячейках. Это сводка повторов тех
же записей, без независимых испытаний. Наибольший рост ошибки в отдельной
средней ячейке относительно корректной метки:

{table(worst[['method','true_age_ms','offset_ms','mae_deg','paired_delta_from_zero_deg']])}

ARX сменил ключ заранее выбранных весов в {changed} из
{len(nonzero_routes)} непустых сравнений «запись, ось, задержка, смещение».
Остальной эффект может возникать и при том же ключе: обе модели используют
метку в построении прогноза. Смена ключа сама по себе не доказывает причину
роста или падения MAE. Все ключи, ошибки по горизонтам и проигрыши сохранены.

Отдельно проверен известный контрпример: истинная задержка 250 мс и горизонт
100 мс. `primary_losses` — число проигранных физических сценариев из семи.
При этой истинной задержке положительные смещения выходят за контракт.

{table(short)}

## Что покрывает сам разброс метки

Дополнительная описательная проверка: взяли минимум и максимум прогноза на
каждом горизонте при всех допустимых метках внутри ±20 мс. Считаем среднюю
ширину этого диапазона, долю измерений внутри и среднее расстояние измерения
за его пределами (ноль, если оно внутри). Ниже одинаковые внутренние
задержки, где доступны все смещения:

{table(display_env)}

Этот диапазон построен вокруг известной в эксперименте истинной задержки.
Он показывает чувствительность к метке и не является откалиброванным
интервалом предсказания или правилом управления.

Проверены {p['paired_angle_history_batches']} парных пакетов и
{p['zero_offset_vectors_checked']} векторов при верной метке. Максимальное
расхождение с прежним replay: {p['maximum_zero_offset_error_rad']:.3g} рад.
Прогнозы и цели сохранены для всех условий, включая проигранные. Сверка
независимо восстанавливает цели и метрики из архивных массивов.

Это проверка постоянной ошибки в метке на старых данных. Из неё нельзя
вывести допуск на реальные часы, устойчивость замкнутого контура или
независимое преимущество. Основной протокол требует подтверждённого возраста
измерения или отдельной оценки по времени синхронного съёма.
''')
    write(OUT/'report.json',dict(independent_confirmation=False,hardware_success_claimed=False,
        final_goal_achieved=False,source_provenance_sha256=sha256(RUN/'provenance.json'),
        verification_sha256=sha256(AUDIT),document_sha256=sha256(DOC),overall_cells=len(dif),
        overall_losses=len(losses),route_key_changes=changed,nonzero_route_rows=len(nonzero_routes),
        common_support_mae=common.to_dict('records'),worst_paired_changes=worst.to_dict('records'),
        short_horizon_at_250ms=short.to_dict('records'),
        envelope_common_support=display_env.to_dict('records')))
    print(common.to_string(index=False))


if __name__ == '__main__':
    main()
