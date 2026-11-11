# Causal regime routing and constrained expert fusion

Preregistered after all prior active-study scores were visible. All33 records are
already open: every new score is retrospective development, never a fresh test.
The prior study improved drive/length-active forecasts but worsened quiet ones.
Here retain the latter exactly, then test selection within active windows.

Inspiration: [Masti, Bernardini and Bemporad2021](https://arxiv.org/abs/2103.12324)
combine observer-bank features with learned virtual sensors for parameter-varying
systems. This motivates a causal model bank. Our constrained forecast fusion is
a different small estimator, not their trained architecture, a reproduced result,
or a stability proof for switching controllers.

## Frozen bank and common information

Do not refit any of the previous125ARX/5forced priors. Load the same development
selection frozen before prior old-evaluation replay. Four current forecasts:
oldARX, active-trained selectedARX, free modal, linear/adaptive forced modal.
For each obtain a3s-prefix forecast of the last already observed1s: six heads
.1,.2,.35,.5,.75,1s; prefix ends at−1s, nominal length and forcing history stop
there too. ARX uses the respective already frozen3s models. Neither target
after0s nor a realised future pose/length/reference is an inference input.

No recurrent expert-memory advantage is introduced: every packet is independent
and uses the same4s history. No time-since-record-start or scenario identity is
a routing feature. All post4s origins and old roles/stride/horizons are retained.

Context = max past401sample x/J range divided by the existing2mm/2mrad cutoff.
L changes remain separately reported, not used to tune a new threshold. Context
≤1 retains legacy bit for bit. A fixed hard router uses prior selected active
blend outside it. A ramp router linearly transitions at context1–2.

Additional active-only weights over four raw forecasts: inverse mean absolute
past error squared plus encoder variance; per-head inverse errors(last1s error
copied for1.5s head); lowest past mean error; and four constrained ridge fits.
The constrained fit minimizes mean_s((B_s−observed_s)w)^2 + lambda||w−prior||²
subject to w≥0 and sum(w)=1. Prior is the hard-context old/new blend weights;
lambda=strength*(mean(all past error²)+quantum²/12), strength .01,.1,1,10.
Enumerate15nonempty faces exactly; independently test against SciPy constrained
optimization, including zero/identical experts. No Bayesian probability or
closed-loop stability is inferred from these weights.

All guarded variants override the quiet forecast with the original value exactly.
Include old/new blends in the11-choice validation table. Feasibility requires
exact legacy equality on every quiet VALorigin; hence an unguarded active method
will usually be infeasible. One global minimum balancedVALscore among feasible
methods, equal fivecases and equal quiet/drive groups within eachcase. Legacy
wins exact ties by declared order. Gate:≥3%better than legacy,≥10%better than
activeARX and≥4/5casewins vslegacy. Save all candidates, weights and failures.
Freeze committed selection/config/weights before this stage's old-evaluation.

## Locked follow-up qualification

After selection freeze, score all12axis-records of the prior7evaluation IDs.
No reselection. Keep original strongARX and activeARX, legacy/active blends and
each router as comparators; report regime/scenario/head metrics and worst losses.
Counts and aggregation distinguish equal-case balanced score from equal-scenario
MAE. Overlapping origins are not independent trials; no new bootstrap claim.

Three synthetic stress conditions, not actual new sensor measurements: independent
uniform angle-history jitter ±quantum/2 or±quantum, and one50ms angle lag with
oldest sample held. Apply the same perturbation to every expert, use original
future measured encoder angles only for scoring; L/support/reference histories
stay identical. Seed fixed by condition/case/record, never chosen by error.
No fitting or routing selection on stress scores. Report unperturbed-context
regimes for comparison; the API always derives its context from supplied history.

Build a separate data-free frozen bundle/API only after development verification,
without changing previous APIs/defaults. Input(N,401) raw100Hz newestfirst,
axis/L/support/native-reference histories; no future-plan argument. Validate
shapes/finite values/units contract. Portable replay must use no dataset or
ancestral run tree. Test15zero-input probes and measure CPU latency, without
claiming exact equilibrium or real-time controller guarantees. Independent
confirmation still requires new suitable author/stand recordings.
