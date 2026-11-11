"""Known timestamp-age qualification, matched references and retained negative results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256
from scripts.run_latency_observer import RUN,OUT
from scripts.report_drive_conditioned import balanced_cases
from scripts.report_angular_control import table
from scripts.run_external_benchmark import verify_hashes

def main():
    if (OUT/'report.json').exists():raise FileExistsError('Preserve earlier report')
    p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());s=json.loads((RUN/'selection.json').read_text());v=json.loads((OUT/'verification.json').read_text());dv=json.loads((OUT/'development_verification.json').read_text());cv=json.loads((OUT/'reference_challenge/verification.json').read_text());assert v['passed'] and dv['passed'] and cv['passed'] and cv['primary_selection_unchanged'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');m=pd.concat([pd.read_csv(RUN/'retrospective/metrics.csv'),pd.read_csv(OUT/'reference_challenge/retrospective/metrics.csv')],ignore_index=True);valid=m[m.windows>0];axes=valid.groupby(['age_ms','noise_quantum','group','case','scenario','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),windows=('windows','first'));records=axes.groupby(['age_ms','noise_quantum','group','scenario','method'],as_index=False).mae_deg.mean();summary=records.groupby(['age_ms','noise_quantum','group','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),scenarios=('scenario','nunique'));axes.to_csv(OUT/'axis_record_summary.csv',index=False);records.to_csv(OUT/'record_summary.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False);allm=summary[summary['group']=='all'];allm.to_csv(OUT/'age_noise_comparison.csv',index=False);fifty=summary[(summary.age_ms==50)&(summary.noise_quantum==0)];fifty.to_csv(OUT/'age50_clean_comparison.csv',index=False);curve=allm[allm.noise_quantum==0].pivot(index='age_ms',columns='method',values='mae_deg').sort_index();curve.to_csv(OUT/'clean_age_curve.csv');balanced=[];cases=[]
    for (age,noise),frame in m.groupby(['age_ms','noise_quantum']):
        c,_=balanced_cases(frame);c['age_ms']=age;c['noise_quantum']=noise;cases.append(c)
        for method,value in c.groupby('method').mae_deg.mean().items():balanced.append(dict(age_ms=int(age),noise_quantum=float(noise),method=method,balanced_mae_deg=float(value)))
    pd.DataFrame(balanced).to_csv(OUT/'balanced_summary.csv',index=False);pd.concat(cases,ignore_index=True).to_csv(OUT/'balanced_case_summary.csv',index=False);val=pd.DataFrame([dict(method=key,balanced_mae_deg=value,positive_age_mae_deg=s['positive_age_means'].get(key),selected=key==s['selected']) for key,value in s['validation_means'].items()]);val.to_csv(OUT/'validation_summary.csv',index=False);refs=pd.DataFrame([dict(case=name,reference_type=case['reference']['type'],reference=str(case['reference']),augmented_model=case['augmented_file']) for name,case in s['cases'].items()]);refs.to_csv(OUT/'reference_choices.csv',index=False);pairs=valid.pivot(index=['age_ms','noise_quantum','group','case','scenario','horizon_s'],columns='method',values='mae_deg').reset_index();pairs['change_vs_matched_arx_deg']=pairs['selected']-pairs['matched_arx'];pairs['change_vs_timestamp_hold_deg']=pairs['selected']-pairs['timestamp_hold'];pairs.to_csv(OUT/'all_paired_changes.csv',index=False);fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for j,noise in enumerate([0,1]):
        frame=allm[allm.noise_quantum==noise].pivot(index='age_ms',columns='method',values='mae_deg').sort_index()
        for method,label,color in [('unaware','Age ignored','#9caaa9'),('timestamp_hold','Timestamp shift + hold','#648b6d'),('selected','VAL-selected adapter','#bb8247'),('matched_arx','Global ARX reference','#3f7183'),('conditional_arx','Late age-conditioned ARX','#735c83')]:axs[j].plot(frame.index,frame[method],marker='o',markersize=4,label=label,color=color)
        axs[j].set_xlabel('Known angle measurement age (ms)');axs[j].set_ylabel('MAE (degrees), equal old scenarios/axes/horizons');axs[j].set_title('Clean history' if noise==0 else 'Synthetic angle-history jitter ±q');axs[j].grid(alpha=.2)
    axs[0].legend(frameon=False);fig.suptitle('Timestamp-aware retrospective sensitivity; 30/70/150ms unused for selection');fig.savefig(OUT/'latency_observer.png',dpi=170);plt.close(fig);selected=s['selected'];c50=curve.loc[50];change50=100*(c50['selected']/c50['unaware']-1);vsref50=100*(c50['selected']/c50['matched_arx']-1);vshold50=100*(c50['selected']/c50['timestamp_hold']-1);explanation='Победил простой timestamp shift + hold. Усложнение физического observer по validation не оправдалось.' if selected=='hold' else 'Выбранный адаптер сравнивается также с простым timestamp shift + hold; результат не сводится к сравнению только с игнорированием меток времени.';g=p['config']['development_gate'];failure=[]
    if s['positive_age_means'][selected]>g['maximum_unaware_positive_age_ratio']*s['positive_age_means']['unaware']:failure.append('gain vs unaware below20%')
    if s['positive_age_means'][selected]>g['maximum_hold_positive_age_ratio']*s['positive_age_means']['hold']:failure.append('gain vs timestamp-hold below3%')
    if s['validation_means'][selected]>g['maximum_matched_arx_ratio']*s['validation_means']['matched_arx']:failure.append('gain vs matchedARX below5%')
    if s['arx_case_wins']<4:failure.append('fewer than4/5matchedARXcasewins')
    clock=json.loads((OUT/'clock_causality.json').read_text());head=json.loads((OUT/'head_diagnostic.json').read_text());assert clock['passed'] and head['passed'] and not head['used_for_selection'];head_table=pd.read_csv(OUT/'head_diagnostic_summary.csv');head_table=head_table[(head_table.age_ms==50)&(head_table.noise_quantum==0)][['method','current_angle_mae_deg','head_mae_deg']];bundle=Path('artifacts/latency_bundle_20261108');files=sum(f.is_file() for f in bundle.rglob('*'));size=sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file());doc=Path('docs/LATENCY_OBSERVER_2026_11_08_RU.md');timing=pd.DataFrame(v['cpu_timing']);timing.to_csv(OUT/'cpu_timing.csv',index=False)
    doc.write_text(f'''# Датированная компенсация сенсорной задержки

Протокол/исходники доfitting: `{p['git_head']}`. Все33исходные записи уже открыты;
это ретроспективный synthetic-age/noise эксперимент. Age дан каждому методу
из условныхtimestamps; неизвестная задержка не оценивалась по будущим ошибкам.

## Контракт и причинность

401измерениеугла заканчивается на current_time−age;401измерение L/support/
native-reference — наcurrent_time. Все100Hz. Началоorigins4.25s,stride1s:
длямаксимальногоage150ms существует настоящая предыстория, безpadding.
Не сравнивать эти matchedчисла напрямую с прежним .25s/padded50msstress.
Каждый считавшийсяorigin сохранён, ноль target/amplitude-basedисключений.
Цели семь раз после текущего момента(.10..1.50s), не послеsensorclock.

Наблюдённыйугол сначала переставляется на свойtimestamp. Только недостающая
новейшаяheadчасть предсказывается: hold, linear.10/.25s,quadratic.25s,
free-modal3s,forced3sfixed/adapt. Forcedobserver подгоняет состояние на датированных
углах и синхронном прошломsupport, затем распространяет его черезgap по
support, уже измеренному к currenttime. Эти supportданные не из будущего
относительноforecastorigin. Нелинейныеnuisanceforceканалы и effectivefrequency
на короткомgap удерживаются; это ограниченная физическая аппроксимация.

Предсказаннаяheadчасть никогда не выдаётся заpasttruth длявесов. Из causal
backtest исключаются головы с timestamp новее последнего полученногоугла:
пять validheads приpositiveage,шесть приage0. Приage0 каждый адаптер и новый
API точно совпадают с прежним frozenregimepredictor. Это не доказательство
аппаратнойустойчивости. Длясвободногоcos(w(t−d)+phi) наблюдается phi−w*d,
а неphiиdотдельно: unknownage изодногоугла вобщемслучае неидентифицируем.

## Сильный соперник

120ARXfits на техжеTRAINrecordings: angle/ioLPV × direct/dated-modal-residual
×6ridge,5cases. Теже синтетическиеages0/20/50/100ms и jitter0/±q.
Известныйage,phase sin(w*age),cos(w*age)−1 и взаимодействияистории — входы
ARX; никакого trialID/noiselabel/target вfeatures. ScalingTRAIN-only,
равныеrecordиquiet/driveвеса. Datedmodalbaseline предсказываетh+age.

ПоVAL percase выбран сильнейший из новых ARX и обоих старых ARX под всеми
темижeadapters. Это сравнение не даёт timestamps только нашемуметоду.
Frozenbank не переобучался и сохраняет прежние TRAINfits на более плотных
origins; новыйaugmentedARX использует предзарегистрированную1sсетку. В пуле
matchedreference остаются теже плотные frozenARX, что и вbank. Это сравнение
полных pipelines с доступом к темжеTRAINrecordings; разницу с augmentation
не следует интерпретировать как отдельный доказанный эффект архитектуры.

{table(refs)}

## Замороженный выбор по validation

Выбран **{selected}**. Developmentgate: **{s['development_gate_passed']}**;
casewins противmatchedARX {s['arx_case_wins']}/5. Невыполненные критерии:
{', '.join(failure) if failure else 'нет'}. Ни один old/stressscore не менялвыбор.

{table(val)}

Gate и этот VALвыбор относятся к исходному globally-selectedARXreference.
Поздний age-conditionedchallenge не менял ни gate, ни выбранныйадаптер.

{explanation}

## Диагностика восстановления угла

После freeze дополнительно оценена самаheadчасть на всех положительныхVALages,
шумах и адаптерах. Это поздний diagnostic, не новыйкритерий выбора.
При50ms чистогоVAL, равныйвеспятиcases:

{table(head_table)}

Истинный дляsyntheticgap исходныйencoderangle использовался только как
offlineцелевой сигнал этойдиагностики. Его не было вinference или selection.
Clockpoisoning ({clock['probes']}probes): замена unreceivedуглов и всех каналов
послеcurrentorigin на большие фиктивныезначения не изменила входы/прогноз.
Ошибка восстановления относительноencoder не является физическимgroundtruth.

## Старые оценочные записи

При известномage50ms и чистойистории: unaware {c50['unaware']:.6f}°,
selected {c50['selected']:.6f}° ({change50:.2f}%), timestamp-hold
{c50['timestamp_hold']:.6f}° ({vshold50:.2f}%vsit), matchedARX
{c50['matched_arx']:.6f}° ({vsref50:.2f}%vsit). Снижение ошибки относительно
игнорированияtimestamps само по себе не доказывает новую physicsnovelty.

{table(curve.reset_index())}

ConditionalARX — дополнительный позднийконтроль после просмотра исходной old
кривой, явно post-hoc. Длякаждогоcase/TRAINage0/20/50/100ms он выбирает
лучшиеARX/adapter поVAL, усредняяобаnoise. Дляновыхages используетсяближайший
TRAINage (tie→меньший). Его20решений закоммичены доегоотдельногоreplay,
никакого новогоTRAINfitting или измененияосновноймодели. При50ms этот
соперник имеет {c50['conditional_arx']:.6f}°; всеусловия приведенывыше.
Это делает сравнениестроже, но не возвращает независимость старымданным.

Режимы при50ms:

{table(fifty)}

Агрегация: равныеheads,оси внутриrecord,затем7records; разныеregimes имеют
разное покрытие. Всеcondition/head/caseпотери и выигрыши вall_paired_changes.
Balanced-score в отдельнойтаблице используетquiet/drive поровну и пятьcases;
его не смешивать с этими equal-scenario MAE. Overlapping origins неtrials.
30/70/150ms отсутствовали вTRAIN/VAL; это новые numericalconditions на старых
данных, не новые независимые измерения. Jitterпакетныйuniform±исходныйencoderq,
будущаяцель — прежнийencoderangle; не модель реального sensornoiseпроцесса.

## Проверки и отдельный профиль

235unit testsPASS доfitting. Независимая matrix-exponential проверка FOH с
двумя шагами времени; tests нулевогоage, genuineprehistory и truthmasking.
ИзTRAIN повторены5выбранныхaugmentedARXfits; все120VALscores, все8адаптеров,
selection/gate и метрики пересчитаны ({dv['forecast_comparisons']}comparisons).
Полныйraw/APIreplay {v['forecast_comparisons']}сравнений; дополнительно
{cv['forecast_comparisons']}replayусловногоARXсвоспроизведением20VALchoices;
все1281main/APHYNITY
оригинала целы, прежниеbundles/defaults не менялись.

Отдельныйbundle `artifacts/latency_bundle_20261108`: {files}files,{size}bytes.
`FrozenLatencyPilot.forecast(...,angle_age_samples=...)` требует известный
целыйage0..15samples. Меткивремени не из будущихtargets. Inference проверен
с запретом np.loadвнеbundle: raw/ancestralruns ему не нужны; нуженmainrepo+deps.
Experimentalprofile сохранён даже приgatefailure, не продвинут вneuraldefault.

{table(timing)}

15zero-inputprobes(L=.4m,5cases×age0/50/150ms): maxbias
{v['maximum_zero_bias_deg']:.6f}°, не exactzero.
Latency — наблюдениеCPUнаданномMac, не real-time/hardwareguarantee.
Новая независимая запись, еёclock/units/encoderresolution и заранее frozen
протокол всёещё обязательны дляисходнойцели. Nativeactuator/MPC не доказаны.
''');r=dict(document=str(doc),document_sha256=sha256(doc),source_sha256=sha256(Path(__file__)),verification_sha256=sha256(OUT/'verification.json'),selected=selected,development_gate_passed=s['development_gate_passed'],validation_gate_failures=failure,age50_unaware_mae_deg=float(c50['unaware']),age50_selected_mae_deg=float(c50['selected']),age50_matched_arx_mae_deg=float(c50['matched_arx']),age50_conditional_arx_mae_deg=float(c50['conditional_arx']),age50_hold_mae_deg=float(c50['timestamp_hold']),age50_change_vs_unaware_percent=float(change50),age50_change_vs_matched_arx_percent=float(vsref50),bundle_files=files,bundle_bytes=size,independent_confirmation=False,final_goal_achieved=False);(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print(curve.to_string())
if __name__=='__main__':main()
