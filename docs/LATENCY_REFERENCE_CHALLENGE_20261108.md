# Late conditional-ARX reference challenge

Added after seeing the original old-evaluation age curve. This is explicitly
post-hoc comparator strengthening on already opened recordings, not independent
confirmation or part of the original development gate. The primary adapter,
original reference, bundle and all earlier predictions remain frozen.

The original reference chooses one ARX/adapter percase across all ages. Allow
a more flexible reference: for eachcase and each TRAIN/VALage0/20/50/100ms,
select the lowest validation quiet/drive-balanced MAE averaged over both noises,
among the same24augmented models and16frozen-ARX/adapter combinations. No new
TRAIN fitting or access to old-evaluation errors for selection. For unseenages
30/70/150ms use nearest TRAINage, lower age wins a distance tie. Neither true
future angle nor noise magnitude is a routing feature.

Commit these20choices before evaluating this extra comparator. Keep every
condition/head/record, report alongside the original globally selected reference,
never replace the primary method by this bank after old-evaluation scores.
Replay all predictions/metrics and reproduce validation choices before reporting.
