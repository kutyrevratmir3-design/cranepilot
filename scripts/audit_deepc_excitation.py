"""Exploratory input-only Hankel ranks on frozen TRAIN records, no model fit.

The depths are provisional and do not implement CranePilot's multirate packet
contract or establish the unknown system order/lag. No matrices cross scenario
boundaries. The union is for reporting only; each fold retains its own TRAIN IDs.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from cranepilot.data import sha256


DEPTHS = (39,45,51)
RELATIVE_THRESHOLDS = (.001,.01)
SUBSAMPLE = 5


def input_hankel(sequence, depth):
    x=np.asarray(sequence,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or not 1<=depth<=len(x):
        raise ValueError('Finite one-dimensional input and admissible depth required')
    return np.lib.stride_tricks.sliding_window_view(x,depth).T.copy()


def ranks(values, shape):
    largest=float(values[0])
    machine_tolerance=max(shape)*np.finfo(np.float64).eps*largest
    return {'machine_tolerance':machine_tolerance,
            'rank_machine':int(np.count_nonzero(values>machine_tolerance)),
            'rank_relative_1e_3':int(np.count_nonzero(values>largest*.001)),
            'rank_relative_1e_2':int(np.count_nonzero(values>largest*.01))}


def synthetic_checks():
    x=np.arange(8,dtype=float)
    expected=np.array([[0,1,2,3,4,5],[1,2,3,4,5,6],[2,3,4,5,6,7]],dtype=float)
    np.testing.assert_array_equal(input_hankel(x,3),expected)
    constant=input_hankel(np.ones(100),39)
    singular=np.linalg.svd(constant,compute_uv=False)
    assert ranks(singular,constant.shape)['rank_machine']==1
    rich=input_hankel(np.random.default_rng(21907).normal(size=300),39)
    singular=np.linalg.svd(rich,compute_uv=False)
    assert ranks(singular,rich.shape)['rank_machine']==39
    zero=input_hankel(np.zeros(100),39)
    singular=np.linalg.svd(zero,compute_uv=False)
    assert ranks(singular,zero.shape)['rank_machine']==0
    return {'manual_indexing':True,'constant_rank_one':True,'deterministic_rich_rank_full':True,
            'zero_rank_zero':True,'uses_real_data':False}


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'artifacts/runs/temporal_20261106/provenance.json'
    provenance=json.loads(source.read_text())
    if provenance['status']!='complete':raise ValueError('Completed frozen source required')
    required={'data/processed/dataset.npz','data/processed/manifest.json'}
    guarded={name.replace('\\','/'):digest for name,digest in provenance['guarded_hashes'].items()
             if name.replace('\\','/') in required}
    if set(guarded)!=required:raise ValueError('Both immutable data hashes must be retained upstream')
    for name,digest in guarded.items():
        if sha256(root/name)!=digest:raise ValueError(f'Dataset changed: {name}')
    folds={name:split['train'] for name,split in provenance['config']['folds'].items()}
    ids=sorted({sid for entries in folds.values() for sid in entries})
    output=root/'results/finetune_campaign_20261107/deepc_excitation'
    if output.exists():raise FileExistsError('Preserve the previous exploratory audit')
    output.mkdir(parents=True)
    checks=synthetic_checks();rows=[];spectra=[];accessed=['time']
    # np.load is lazy: access only the time base and xr/L for TRAIN-union IDs.
    # No measured future theta, position, validation-only or evaluation-only
    # record is read, and no model/selection state is opened or modified.
    with np.load(root/'data/processed/dataset.npz',allow_pickle=False) as arrays:
        time=arrays['time']
        if not np.allclose(np.diff(time),.01,atol=1e-10):raise ValueError('Expected10ms source grid')
        for sid in ids:
            accessed.extend([f's{sid}_xr',f's{sid}_L'])
            reference=np.asarray(arrays[f's{sid}_xr'],dtype=float)[::SUBSAMPLE]
            length=np.asarray(arrays[f's{sid}_L'],dtype=float)
            if len(reference)!=len(time[::SUBSAMPLE]) or not np.isfinite(reference).all():
                raise ValueError('Invalid sampled command')
            if np.ptp(length)>=1e-6:raise ValueError('Fixed-length TRAIN record required')
            common={'scenario':sid,'train_folds':'|'.join(name for name,entries in folds.items() if sid in entries),
                    'length_m':float(length[0]),'sample_interval_s':.05,'sample_count':len(reference),
                    'command_min':float(reference.min()),'command_max':float(reference.max()),
                    'command_range':float(np.ptp(reference)),'command_std':float(reference.std()),
                    'command_unique_values':int(len(np.unique(reference))),
                    'command_changes_abs_gt_1e_12':int(np.count_nonzero(np.abs(np.diff(reference))>1e-12))}
            for depth in DEPTHS:
                matrix=input_hankel(reference,depth)
                singular=np.linalg.svd(matrix,compute_uv=False)
                if not np.isfinite(singular).all():raise ValueError('Nonfinite singular spectrum')
                rank=ranks(singular,matrix.shape)
                condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
                if condition is not None and not np.isfinite(condition):condition=None
                row={**common,'depth':depth,'hankel_rows':matrix.shape[0],'hankel_columns':matrix.shape[1],
                     'largest_singular_value':float(singular[0]),'smallest_singular_value':float(singular[-1]),
                     'condition_number_raw':condition,**rank,'full_row_rank_machine':rank['rank_machine']==depth}
                rows.append(row)
                for index,value in enumerate(singular):
                    spectra.append({'scenario':sid,'depth':depth,'singular_index_one_based':index+1,
                                    'singular_value':float(value),
                                    'relative_to_largest':float(value/singular[0]) if singular[0]>0 else 0.})
    table=pd.DataFrame(rows);table.to_csv(output/'rank_summary.csv',index=False)
    pd.DataFrame(spectra).to_csv(output/'singular_values.csv',index=False)
    configuration={'input':'xr in original recorded reference scale; servo scaling unverified',
        'input_preprocessing':'raw sequence sampled with[::5]; no demeaning, shifting, filtering or concatenation',
        'sample_interval_s':.05,'depths':list(DEPTHS),'provisional_depth_interpretation':'9 history +30 future +assumedorders0/6/12; only a probe',
        'machine_tolerance':'max(matrix.shape)*float64_eps*sigma_max','relative_thresholds':list(RELATIVE_THRESHOLDS),
        'known_system_order':None,'known_system_lag':None,'operational_multirate_contract_implemented':False,
        'command_to_output_index_alignment_implemented':False,'model_training':False,
        'affects_current_campaign_selection':False,'exploratory_after_campaign_preregistration':True}
    write(output/'configuration.json',configuration)
    write(output/'verification.json',{'passed':True,'synthetic_checks':checks,'verified_data_hashes':guarded})
    shown=table[['scenario','train_folds','depth','rank_machine','rank_relative_1e_3','rank_relative_1e_2','command_changes_abs_gt_1e_12']]
    text=shown.to_string(index=False)
    full=int(table.full_row_rank_machine.sum())
    verdict=(f'По машинному порогу полный строчный ранг имеют {full} из {len(table)} матриц. '
             f'При относительном пороге1e-3 ранги лежат в диапазоне{table.rank_relative_1e_3.min()}–{table.rank_relative_1e_3.max()}, '
             f'при1e-2 —{table.rank_relative_1e_2.min()}–{table.rank_relative_1e_2.max()}.')
    report=f'''# Разведочная проверка входного возбуждения для DeePC

Проверены только TRAIN-ID хотя бы одного зафиксированного разбиения: {ids}.
{verdict}

Это объединение нужно исключительно для компактного отчёта. При обучении
какого-либо будущего метода каждый fold обязан использовать только собственные
TRAIN-ID: {json.dumps(folds,ensure_ascii=False)}. Сценарий может быть TRAIN одного
fold и evaluation другого; результаты этой разведки не меняют текущий выбор
моделей и не превращают объединение в разрешённый общий тренировочный набор.

```text
{text}
```

Исследован сырой reference xr в исходном масштабе, взятый каждые50мс без
вычитания среднего, сдвига или фильтрации. Матрицы каждого сценария отдельны.
Все сингулярные числа сохранены; малый относительный ранг означает зависимость
числа различимых направлений от масштаба отсечения, а не автоматически провал
теоретического условия. Неизвестны порядок и lag объекта. Подсчёт изменений
команды использует порог абсолютной разности1e-12 в исходной шкале.

Глубины39/45/51 условно связаны с9+30 и добавлением предполагаемого порядка6/12.
Это не реализация DeePC и не доказательство persistent excitation нужного порядка.
Настоящий входной контракт имеет41past×10мс и30future×50мс. Согласование сеток,
текущей команды и выхода, а также достаточность0.4с для состояния ещё не сделаны.
Шум может создавать полный численный ранг; низкий ранг также зависит от DC-уровня
при выбранной нормировке. Нельзя заключать из этой таблицы, что метод обязательно
заработает или что данные бесполезны. Следующий шаг при разработке — проверить
масштабирование и устойчивость спектра по TRAIN-блокам, затем валидировать
ограниченный прогноз на отдельной validation при фиксированном протоколе.

Данные и manifest проверены по исходному temporal provenance до чтения массивов.
Выполнены независимые проверки индексов Hankel, постоянного, нулевого и богатого
синтетического входа. Будущие углы и позиции не читались; оценки модели,
checkpoint и selection не использовались. Эта разведка сделана после регистрации
fine-tuning кампании и не влияет на её обучение или критерий принятия.

Теоретический контекст: [DeePC2019](https://arxiv.org/html/1811.05890v2),
[разбор для CranePilot](../../../docs/CONTROL_DATA_AND_REGULARIZATION_20261107.md).
'''
    (output/'FINDINGS_RU.md').write_text(report)
    write(output/'provenance.json',{'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_provenance':str(source.relative_to(root)),'source_provenance_sha256':sha256(source),
        'script':str(Path(__file__).resolve().relative_to(root)),'script_sha256':sha256(Path(__file__)),
        'train_ids_by_fold':folds,'unique_train_ids_analyzed':ids,'npz_arrays_accessed':accessed,
        'verified_data_hashes':guarded,'configuration':configuration,
        'result_hashes':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    print(verdict);print(text);print('Saved',output)


if __name__=='__main__':main()
