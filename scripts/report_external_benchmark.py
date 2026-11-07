"""Full external-rig result report with separate frozen and locally fitted tracks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256
from scripts.report_angular_control import table

OUT=Path('results/external_benchmark_20261107');RUN=Path('artifacts/runs/external_benchmark_20261107')

def main():
    out=OUT;run=RUN;dest=out/'report.json'
    if dest.exists():raise FileExistsError('Preserve prior report')
    p=json.loads((run/'provenance.json').read_text());v=json.loads((out/'verification.json').read_text());cfg=p['config'];selection=json.loads((run/'selection.json').read_text())
    assert p['status']=='complete' and v['passed'] and v['provenance_sha256']==sha256(run/'provenance.json')
    for mapping,base in [('guarded_hashes',Path('.')),('training_hashes',run),('result_hashes',run)]:
        for name,digest in p[mapping].items():assert sha256(base/name)==digest,name
    allrows=pd.concat([pd.read_csv(run/stage/'metrics.csv') for stage in ['validation','test']],ignore_index=True);allrows.to_csv(out/'all_metrics.csv',index=False)
    records=allrows.groupby(['stage','record','method'],as_index=False)[['mae_deg','rmse_deg','p95_absolute_error_deg']].mean();records.to_csv(out/'record_summary.csv',index=False)
    summary=records.groupby(['stage','method'],as_index=False)[['mae_deg','rmse_deg','p95_absolute_error_deg']].mean()
    tracks={m:('frozen_external_weights' if m.startswith('cranepilot') else 'locally_fitted' if m in ['fitted_oscillator','arx_residual','lstm_residual'] else 'unfitted_reference') for m in cfg['methods']}
    summary['track']=summary.method.map(tracks);summary.to_csv(out/'summary.csv',index=False)
    horizons=allrows.groupby(['stage','method','nominal_horizon_s'],as_index=False)[['mae_deg','rmse_deg']].mean();horizons.to_csv(out/'horizons.csv',index=False)
    clock=pd.concat([pd.read_csv(run/stage/'qualification.csv') for stage in ['validation','test']],ignore_index=True);clock.to_csv(out/'camera_clock.csv',index=False)
    ev=records[records.stage=='test'];matrix=ev.pivot(index='record',columns='method',values='mae_deg').loc[cfg['test_ids'],cfg['methods']]
    rng=np.random.default_rng(cfg['bootstrap']['seed']);ids=rng.integers(0,4,(cfg['bootstrap']['draws'],4));comparison=[]
    for reference in ['geometric','fitted_oscillator','arx_residual','lstm_residual']:
        r=matrix[reference].to_numpy()
        for method in cfg['methods']:
            c=matrix[method].to_numpy();change=100*(c.mean()/r.mean()-1);effect=100*(c[ids].mean(1)/r[ids].mean(1)-1)
            comparison.append(dict(method=method,reference=reference,change_percent=change,record_wins=int((c<r).sum()),records=4,descriptive_low_percent=float(np.quantile(effect,.025)),descriptive_high_percent=float(np.quantile(effect,.975))))
    comparisons=pd.DataFrame(comparison);comparisons.to_csv(out/'paired_record_comparisons.csv',index=False)
    timing=[]
    # Training duration is diagnostic, not wall-clock benchmarking across methods.
    for row in selection['lstm']:timing.append(row)
    pd.DataFrame(timing).to_csv(out/'lstm_checkpoints.csv',index=False)
    pivot=summary[summary.stage=='test'].set_index('method');frozen=pivot.loc['cranepilot_retained','mae_deg'];geometric=pivot.loc['geometric','mae_deg'];lstm=pivot.loc['lstm_residual','mae_deg']
    frozen_effect=100*(frozen/geometric-1);lstm_effect=100*(lstm/geometric-1)
    fig,axes=plt.subplots(1,2,figsize=(14,5.2),layout='constrained');methods=cfg['methods'];colors=['#89939a','#afb4b8','#29758c','#25846e','#a57a43','#485c98','#a65c6c','#bd9254']
    axes[0].bar(methods,[pivot.loc[m,'mae_deg'] for m in methods],color=colors);axes[0].set_ylabel('MAE (deg), equal records and seven horizons');axes[0].tick_params(axis='x',rotation=50,labelsize=8);axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
    for m,color in zip(methods,colors):
        sub=horizons[(horizons.stage=='test')&(horizons.method==m)];axes[1].plot(sub.nominal_horizon_s,sub.mae_deg,label=m,color=color)
    axes[1].set_xlabel('Nominal sample-index horizon (s)');axes[1].set_ylabel('MAE (deg)');axes[1].legend(fontsize=7);axes[1].grid(alpha=.2);fig.suptitle('External rotary crane: 4 sealed real-camera records; frozen and locally fitted tracks');fig.savefig(out/'external_comparison.png',dpi=170);plt.close(fig)
    # Full time-index forecast error distributions, no selection of a favorable example.
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,sid in zip(axes.flat,cfg['test_ids']):
        sub=ev[ev.record==sid].set_index('method');ax.bar(methods,[sub.loc[m,'mae_deg'] for m in methods],color=colors);ax.set_title(f'Record {sid}');ax.set_ylabel('MAE (deg)');ax.tick_params(axis='x',rotation=55,labelsize=7);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'all_test_records.png',dpi=150);plt.close(fig)
    doc=Path('docs/EXTERNAL_BENCHMARK_2026_11_07_RU.md');diagnostics=json.loads((out/'posthoc_diagnostics.json').read_text())
    # All displayed numerical comparisons derived directly from the frozen run.
    text=f'''# Независимый стенд: причинный прогноз по raw camera, 7 ноября 2026

Впервые проверен прогноз на записях **другого лабораторного крана**, ранее не
использованных в CranePilot. Протокол/код зафиксированы в `{p['git_head']}` до
обучения. TRAIN12 records, VALIDATION4, закрытый TEST4 (17–20); все8методов
проверены на каждой записи. Source branch: research/broad-search-20261107.

## Основной результат

**Замороженный CranePilot не подтвердил перенос на этот стенд.** Он проигрывает
геометрическому baseline на всех4закрытых records и хуже даже zero/persistence
по общей MAE. В этом сравнении минимальная средняя ошибка у ARX residual:
{pivot.loc['arx_residual','mae_deg']:.6f}°. Это обычная линейная residual model,
а не достижение новой нейронной архитектуры. ARX выигрывает у fitted oscillator
на всех4records, примерно на52.24% по среднему. Neural candidate тоже не спасает
перенос; выбор между frozen profiles по TEST не делался.

Frozen retained CranePilot MAE **{frozen:.6f}°**, geometric **{geometric:.6f}°**:
изменение ошибки **{frozen_effect:+.2f}%**. Это проверка замороженных угловых весов
после заранее заданной геометрической/частотной адаптации и TRAIN-only centering,
не строгое zero-calibration применение полной native pipeline.

Locally fitted LSTM residual MAE **{lstm:.6f}°**, изменение против geometric
**{lstm_effect:+.2f}%**. У него есть12новых тренировочных записей, у frozen angular
weights их нет: этот разрыв нельзя приписывать только качеству архитектуры.
Это predictive error, не снижение sway реальным контроллером. Ни один default
или bundle weight не изменён; дальнейшего обучения после TEST не было.

## Все результаты закрытого TEST

{table(summary[summary.stage=='test'].drop(columns='stage'))}

Среднее равновесно по четырём records, затем по семи horizons. RMSE и p95 здесь
средние record×horizon диагностик, НЕ единый pooled RMSE/p95 всех окон.

{table(matrix.reset_index())}

Сравнения и число выигранных RECORDS (не overlapping windows):

{table(comparisons)}

Bootstrap resamples4whole-record clusters,2000draws,seed7112027,одни индексы для
всех методов. Интервалы описательные: четыре records одного троса/груза не
дают точную оценку переноса на популяцию кранов. Все horizons/records/negatives
сохранены. По TEST не выбирались коэффициенты, seed, epoch или preprocessing.

## Новый источник и обработка

[Khemakhem et al., arXiv2605.14944](https://arxiv.org/abs/2605.14944),
[публичный author repository](https://github.com/ikhemakhem/BehavioralTrajGenCrane),
commit `{cfg['commit']}` от {cfg['commit_date']}. README упоминает раннюю2024версию,
preprintv1 опубликован14мая2026 и помечен under review; статус рецензирования
не предполагается. Записи физического поворотного крана с камерой nominal20Hz,
rope1m. Авторский validation_data разделён НАМИ на12/4/4 целых records по IDs;
их29processedTRAINrecords не использовались.

Одна TRAIN record1 и calibration были просмотрены для schema qualification до
preregistration, без model fitting/prediction. Остальные test17–20не разбирались
до lock выбора. Копирование/проверка двоичных файлов по Git SHA не было просмотром
углов. Download manifest содержит Git-blob SHA1, SHA256, URLs/размеры41MATфайла.
Сами внешние raw/source не републикуются: fetch script получает pinned source.

Author pipeline использует filtfilt и mean по ВСЕЙ записи. Мы независимо
преобразуем raw pixels в углы с фиксированной camera calibration на каждом frame;
не используем ни filtfilt, ни per-test mean, ни author vision_data, ни simulated
input-file yd. Масштаб1.7 взят из source preprocessing, не fitted по нашему TEST.
Один constant offset={selection['offset_rad']:.9f}rad — equal-record TRAIN mean.
Это калибровочный выбор, не доказательство физической точности камеры1.7.

Общий контракт:9native samples(theta2,boom encoder,issued profile) over nominal.4s;
30future prescribed velocity/acceleration steps. Будущие measured x/theta не
входы. Raw поля velocity/acceleration совпадают с prescribed input profile;
они не объявляются независимыми сенсорными измерениями. Углы radial theta1
не используются ни как вход, ни как выход — это tangential-only benchmark.

Frozen input adapter x=R*boom_angle, a=R*planned_acceleration — локальное
приближение trolley координаты. Coupling/centrifugal terms из rotary dynamics
не моделируются: перенос сложнее исходного trolley task. Omega=sqrt(g/1m) по
метаданным нового стенда; старая Inteco length calibration не переносится.
Нативный xr=0, старый NeuralActuator bypass. Из9nativepoints получены41samples
путём previous-sample hold; −10ms берёт−50ms, без будущей интерполяции. Новая
информация при upsampling не возникает. Это адаптированный angular submodel,
не перенос полной native-command CranePilot.

## Часы камеры и горизонт

Task по sample index, nominal dt=.05s. Истинные frame timestamps нерегулярны;
мы не подавали будущие timestamps модели. Поэтому1.5s — nominal30steps, а не
подтверждённый exact wall-clock horizon. Длительности реальных окон:

{table(clock)}

Range/median каждого фактического horizon находятся в all_metrics.csv (для
каждого record, без поиска лучших сдвигов). Physical command-camera timing
проверен только через supplied index alignment, не независимой аппаратурой.

## Диагностика причин, без изменения предикторов

Средний TRAIN camera interval {diagnostics['train_mean_camera_interval_s']:.9f}s
вместо nominal.05s. Fitted omega {diagnostics['fitted_omega_per_nominal_second']:.6f}
на nominal clock после ДИАГНОСТИЧЕСКОГО пересчёта по TRAIN ratio соответствует
{diagnostics['clock_rescaled_fitted_omega_rad_per_actual_second']:.6f}rad/s;
geometric omega={diagnostics['geometric_omega_rad_per_second']:.6f}rad/s. Это позволяет
предположить, что часть fitted frequency shift связана с timebase, а не только
с неизвестной механикой. Скорректированные прогнозы здесь НЕ строились и не
выбирались; нельзя объявлять clock drift доказанной единственной причиной.

На отдельном synthetic zero-history/zero-command packet frozen retained выдаёт
до {diagnostics['zero_packet_peak_deg']['cranepilot_retained']:.6f}°, candidate до
{diagnostics['zero_packet_peak_deg']['cranepilot_candidate']:.6f}°, geometric ровно0.
Это posthoc проверка математического equilibrium, не аппаратная ошибка на реальном
test и не количественное разложение его MAE. Она показывает, что текущая learned
correction не гарантирует нулевой отклик при нулевом входе. Camera centering,
upsampling и геометрический domain adapter также меняют распределение входов.
На этом основании ни один weight/offset/preprocessing не исправлялся после TEST.

## Базовые методы и связь с публикациями

Geometric — forced observer+ZOH oscillator с геометрической частотой. PEM grey-box
fitted_oscillator имеет omega,damping,gain, fitted по TRAIN multi-step output error;
initial state профилируется по9pastangles/issuedacceleration в каждом origin.
Три train-only startingpoints, bounds в config; validation не выбирает параметры.
Финальные параметры {selection['oscillator_parameters']}.

ARX residual — direct linear command-prefix prediction поверх geometric, ridge
alpha={selection['arx_alpha']}, выбранный на VALIDATION из3значений. Constant length
в этом dataset не позволяет проверить LPV scheduling; метод не назван SMILE.
LSTM residual — один32unit encoder/decoder,3seeds,zero-output epoch−1 baseline,
early stop по VALIDATION, equal-record seven-horizon MSE training. Selection:

{table(pd.DataFrame(selection['lstm']))}

[Zhou2025 PINN](https://doi.org/10.1016/j.aei.2025.103269) и
[CoDIT2026 LPV](https://doi.org/10.1109/CoDIT70676.2026.11631165) подтверждены по
аннотациям; полного проверенного code/loss/architecture нет. Наши результаты
НЕ репликация их published scores. ARX/multistep identification мотивированы
[Kusznir/Smoczek2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8749516/),
а сравнительный набор recurrent/classical methods — также
[Kusznir et al.2024](https://www.mdpi.com/2076-3417/14/8/3492). Их genetic search
не повторяется. Методика author behavioral trajectory optimization в этом этапе
не реализована: goal — общий короткоисторический причинный прогноз, их task/input
history и acausal offline filtering отличаются. Нельзя сообщать, что мы обошли
эти статьи по их собственному протоколу.

## Проверки и воспроизведение

179unit tests PASS: geometry scalar/vector equivalence,source corruption,partition
seals,futurepixelprefix poisoning,known-system PEM recovery,ARX/LSTM commandprefix,
frozen target invariance и unchangedweights. Все {v['forecast_trajectories']}
validation+TEST forecasts replayed из сохранённых checkpoints/оригинальных raw,
а не из saved estimates. Семь metric horizons всех records пересчитаны независимо.
SHA guarded source/data/checkpoints/selection проверены, все
{v['original_files']['checked_files']} original main+APHYNITY файлов сохранены.
Raw original cache excluded from public artifacts. APHYNITY без изменений.

Новый запуск: fetch_external_crane,run_external_benchmark --stage train, затем
--stage evaluate; verify_external_benchmark,report_external_benchmark. Все через
restored main Python3.12launcher. Scripts запрещают overwrite. Для повторения
benchmark нужно новое --output и согласованные RUN/OUT paths report/verifier.
Artifacts/runs копируется в results/external_benchmark_20261107/frozen_run,
включая source manifest, все fitted weights и численные predictions. Сами raw
сигналы скачиваются заново по pinned manifest, не вложены в evidence ZIP/Git.

## Что это доказывает и что остаётся

Есть действительно новый для проекта стенд и закрытая record-level проверка
прогноза. Есть общие причинные inputs и сильные locally fitted comparators,
все отрицательные переносы сохраняются. Нет нового hardware control опыта,
нет доказательства стабильности/SOTA, нет переносимости на новые длины/грузы
по одному fixed1m dataset. Frozen/local training access различается явно.
Native xr semantics старого стенда по-прежнему не установлены.
После TEST этот набор уже нельзя снова называть нетронутым подтверждением
следующей настроенной модели. Любая адаптация CranePilot потребует нового
confirmation set или явной маркировки retrospective ablation.
'''
    doc.write_text(text)
    report=dict(document=str(doc),document_sha256=sha256(doc),report_source_sha256=sha256(Path(__file__)),provenance_sha256=sha256(run/'provenance.json'),verification_sha256=sha256(out/'verification.json'),aggregation_helper_sha256=sha256(Path('scripts/report_angular_control.py')),posthoc_diagnostics_sha256=sha256(out/'posthoc_diagnostics.json'),external_test_records=4,external_real_predictions_verified=True,external_full_controller_validated=False,exact_paper_reproduction=False,frozen_retained_change_vs_geometric_percent=frozen_effect,lstm_change_vs_geometric_percent=lstm_effect)
    dest.write_text(json.dumps(report,indent=2)+'\n');print(summary.to_string(index=False));print('Frozen vs geometric %',frozen_effect)

if __name__=='__main__':main()
