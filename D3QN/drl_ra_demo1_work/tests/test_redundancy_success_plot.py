from plot_redundancy_success import aggregate_records, wilson_interval


def test_aggregate_records_groups_empirical_success_by_replica_count() -> None:
    records = [
        {"replicas": 1, "reliability_success": 1},
        {"replicas": 1, "reliability_success": 0},
        {"replicas": 2, "reliability_success": 1},
        {"replicas": 2, "reliability_success": 1},
    ]

    result = aggregate_records(records)

    assert result["1"]["tasks"] == 2
    assert result["1"]["successes"] == 1
    assert result["1"]["success_rate_pct"] == 50.0
    assert result["2"]["success_rate_pct"] == 100.0


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(8, 10)

    assert lower < 80.0 < upper
