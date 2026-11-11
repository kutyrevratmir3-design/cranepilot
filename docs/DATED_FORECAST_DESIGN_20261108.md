# Direct dated forecasts, measured-gap propagation and changing length

Preregistered after all previous latency scores were visible, including the
slight dated-modal advantage at150ms. All33physical recordings already opened:
this is retrospective synthetic sensitivity work, not independent confirmation.

## Fixed input contract and comparison grid

401angle samples end at current−known_age;401support/L/native-reference samples
end now, all100Hz. Known scalar integerage0..25samples; no unknown-delay inference.
Origins4.25s,stride1s preserve genuine prehistory even at250ms. TRAIN/VALages
0/20/50/100/150/200ms, clean or uniform angle jitter±encoderq. Evaluate old7IDs
also at30/70/250ms. No actual future pose/L/angle as inference inputs. Same
recordroles/horizons as before; no amplitude/error exclusions. No closed-loop
resimulation: recorded support/commands remain fixed during synthetic anglelag.

Primary parent comparator extends the exact previous quadratic(.25s) algorithm
to250ms in a new implementation without changing previous files. At≤150ms its
forecasts must match the frozen parent. Every new candidate preserves original
regime output exactly atage0; bare dated-modal remains a separately reported
baseline, so the zero-age guard cannot hide its different score.

## Direct model bank

Free4s evaluates its fitted modal state at h+age, avoiding invented current angle.
Five further constant-length models fit2.5s of truly observed dated angles:
free modal and forced hold/linear support × fixed/adaptive TRAINcouplingprior.
The forced model synchronizes its past support with angle measurement time,
propagates through the measured historical sensor gap, then through explicitly
assumed future support. Past basis FOH50ms, future/gap FOH10ms. Nonlinear force
channels use backward50ms derivative stencils, not double differentiation of
noisy measured support. Prior coefficients/scales stay frozen; no new physical fit.

One homogeneous changing-length model propagates fitted modal states through
the measured Lgap, then holds current L into future. Effective nominal length
uses the same train calibration. For each mode, stiffness scales as Lsensor/L(t)
and damping adds2*Ldot/L. This is a small-angle approximation with a persistent
fitted offset, not an identified nonlinear3Dcrane theorem. Low-amplitude states
retain their observed mean. Constant L must reproduce the analytical free model.
RK4 at10ms is independently checked against solve_ivp.

Variable-length inspiration: [Belyakov/Seyranian2009](https://arxiv.org/abs/0910.3802),
whose angular-momentum equation yields the length-rate term, and
[Gonzalez-Resines etal2017](https://arxiv.org/abs/1705.06883), on coupled crane
transport/rope-length design. Our local empirical two-mode approximation and
forecast experiment do not reproduce their control or stability results.

## Causal weighting, without estimated-angle truth

The six2.5s models form a direct bank. A shadow packet at current−1s uses
only samples available there and the same known angle age. Its oldest required
support is1+age+2.5≤3.75s old, within the provided4s. Its future support is its
own continuation, never the actually observed later trajectory. Six shadow
head offsets relative to that current clock: .10,.20,.35,.50,.70,1−age seconds.
The last target is the latest received angle, not estimated current angle;
all six targets are truly observed. Constant-length and changing-length models
have identical current/shadow history lengths. No recurrent extra-memory input.

Test inverse past-MAE² weights and simplex-ridge strengths.1/1/10 using the
existing encoder floor. Prior: half-free/half-variableL when pastsupportquiet;
otherwise .25free,.25variableL,.125eachforced. These are heuristics, not
probabilities or a controller guarantee. Also test fixed age-dependent blends
of parent with free4 or bank(simplex1), weight age/(age+tau), tau50/100/200ms;
two horizon-damped variants and one fixed quiet-only free4 blend.21global
candidates. The complete negative grid is retained.

## Stronger ARX, selection and verification

Fit120new age-augmented ARXs on the same TRAINrecordings/origins/conditions:
angle/ioLPV × direct/dated-modal-residual ×6ridge. Add all120old augmented fits
and both frozen ARXs under all8original head adapters, extended consistently
to250ms. Per-case/per-TRAINage reference chosen only by VAL, noise averaged;
unknown evaluation ages use nearest TRAINage, lower age on distance ties.
All methods have the same known age and available histories. No test-based
reference or primary reselection. Existing bank keeps its original denser fits;
new references use1sorigins, with all old dense ARXs still in the reference pool.

One global primary minimum equal-five-case, equalage/noise, quiet/drive-balanced
VALscore. Gate requires≥3%gain vsparent,≥5%vsmatchedARX,≥4/5ARXcasewins and
no TRAINage mean more than5%worse thanparent. Freeze source before fits/selection,
freeze copied models/selection before replay. Save all candidates and cases,
including bare dated-modal and all losses. Replay source/raw-to-forecast, full
metrics, age0identity, historical-gap causality, and data-free API; record CPU
latency/zero bias. Preserve all previous APIs/bundles/defaults and APHYNITY.
New independent author/stand trials are still required for the thread goal.
