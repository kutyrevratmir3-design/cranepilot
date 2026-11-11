# Sensitivity to erroneous angle age metadata

This is a retrospective input-contract stress test. All seven old physical
records were already opened and repeatedly analyzed. It gives no independent
confirmation and does not change the prospective gate or either model.

Construct the same 401-value newest-first histories and origins used in the
previous dense reference replay. Support, length and native references end at
the true origin k. Angles end at k minus the simulated true age. Targets are
the unchanged measured angles at k + h. Use clean histories, with no additional
noise or smoothing. Both frozen APIs get copies of exactly the same histories.

For true ages 0, 2, 5, 10, 15, 20 and 25 raw samples, pass a declared age equal
to true age plus -2, -1, 0, +1 or +2 samples. Omit values outside the supported
0..25 age contract; do not clamp or compensate. One raw sample is 10 ms. A
positive offset tells the API the measured angle is older than it really is.
The current support channels and actual angle samples never move when changing
this metadata. This represents a wrong delay label, not a changed sensor trace.

Score the unchanged CranePilot mixture and the frozen VAL-selected dense ARX.
The ARX can change its predeclared age route; retain the actual key and report
route changes. The primary can change its built-in age weighting. Do not
reselect either model or search over offsets. The zero-offset baseline must
reproduce the archived replay (same origins, route and values to 1e-12 rad).
Every supported age/offset/scenario/axis/horizon and every loss is retained.

Compute MAE per horizon over origins. Also compute all-horizon MAE per axis,
then average axes inside each physical record and average the seven records.
For signed offset effects subtract the corresponding zero-offset error at
identical true age, scenario, axis and horizon. Offset summaries balance only
the true ages supporting that offset; a separate common-support summary
balances the same interior ages for all offsets. No unit of analysis is treated
as an independent new trial and no inference/acceptance threshold is applied.

The test does not simulate time-varying jitter, asynchronous channels or
closed-loop behavior. Good results cannot certify tolerance on a real device.
Bad cases identify where the accurate age requirement matters.
