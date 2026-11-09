"""Reproduce session-design arithmetic with the frozen gate and synthetic MAEs."""
from fractions import Fraction
from itertools import combinations
import json
from math import ceil

from scripts import fresh_confirmation as confirmation
from scripts.fresh_record_guard import sha256_file


def missing_probability(supports):
    """Exact chance that a shared-cluster bootstrap draw omits a condition."""
    groups = set().union(*supports)
    cluster_count = len(groups)
    missing_draws = 0
    for count in range(1, len(supports) + 1):
        sign = 1 if count % 2 else -1
        for omitted in combinations(supports, count):
            remaining = cluster_count - len(set().union(*omitted))
            missing_draws += sign * remaining ** cluster_count
    return Fraction(missing_draws, cluster_count ** cluster_count)


def synthetic_case(name, supports, session_deltas, settings):
    rows = [
        dict(session_id=f'session{group}', trial_id=f'condition{condition}',
             condition_id=f'condition{condition}', independent_cluster_id=f'session{group}',
             candidate_mae_deg=1.0 + session_deltas[group], reference_mae_deg=1.0)
        for condition, groups in enumerate(supports)
        for group in groups
    ]
    scheme = confirmation.hierarchy(rows)
    probability = (missing_probability(supports) if scheme == 'shared_clusters_fixed_conditions'
                   else Fraction(0))
    return dict(name=name, session_ids_by_condition=[list(groups) for groups in supports],
                synthetic_session_delta_deg=session_deltas, synthetic_reference_mae_deg=1.0,
                exact_missing_condition_penalty_probability=dict(numerator=probability.numerator,
                                                                 denominator=probability.denominator,
                                                                 decimal=float(probability)),
                synthetic_gate=confirmation.paired_statistics(rows, settings, scheme))


def main():
    settings = confirmation.policy()
    expected = dict(minimum_conditions=5, minimum_independent_clusters=5,
                    minimum_relative_gain=.10, condition_win_fraction_exclusive=.5,
                    bootstrap_draws=10000, bootstrap_seed=20261108,
                    bootstrap_upper_quantile=.975)
    frozen = {key: settings[key] for key in expected}
    if frozen != expected:
        raise ValueError('This planning report requires the frozen November 8 gate settings')

    sample_count = settings['bootstrap_draws']
    quantile = Fraction(str(settings['bootstrap_upper_quantile']))
    quantile_index = ceil(quantile * (sample_count - 1))
    groups = tuple(range(5))
    fully_crossed = [groups] * 5
    all_wins = [-.2] * 5
    cases = [
        synthetic_case('nested_one_session_per_condition', [(g,) for g in groups], all_wins, settings),
        synthetic_case('fully_crossed_uniform_gain', fully_crossed, all_wins, settings),
        synthetic_case('fully_crossed_adverse_session', fully_crossed, [-.2] * 4 + [.2], settings),
        synthetic_case('fully_crossed_milder_adverse_session', fully_crossed, [-.2] * 4 + [.1], settings),
        synthetic_case('cyclic_three_of_five', [tuple((g + i) % 5 for i in range(3)) for g in groups],
                       all_wins, settings),
        synthetic_case('rotating_four_of_five', [tuple(g for g in groups if g != c) for c in groups],
                       all_wins, settings),
    ]
    report = dict(
        scope='Session-design arithmetic and synthetic MAEs only',
        frozen_policy_sha256=sha256_file(confirmation.POLICY),
        frozen_gate_source_sha256=sha256_file(confirmation.__file__),
        frozen_settings=frozen,
        maximum_missing_draws_for_finite_upper_quantile=sample_count - quantile_index - 1,
        minimum_missing_draws_for_infinite_upper_quantile=sample_count - quantile_index,
        shared_five_session_per_condition_missing_probability={
            count: float(Fraction(5 - count, 5) ** 5) for count in range(1, 6)},
        synthetic_cases=cases,
        fully_crossed_minimum_example=dict(sessions=5, conditions=5, trials=25),
        recommended_acquisition_target=dict(sessions=8, conditions=6, trials_per_cell=1, trials=48,
                                            every_session_covers_every_condition=True),
        power_estimated=False, independent_confirmation=False, final_goal_achieved=False,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
