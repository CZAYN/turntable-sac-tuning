import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from elc_rl.controller_parameters import (
    PARAMETER_ORDER,
    derive_physics_controller_initials,
    load_current_pidf_feasibility_evidence,
    load_physics_controller_parameter_space,
)
from elc_rl.performance_targets import load_controller_performance_targets
from elc_rl.physics_motor_model import load_physics_motor_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _space():
    return load_physics_controller_parameter_space(PROJECT_ROOT)


def test_physics_parameter_order_and_initials_use_the_real_target_table():
    space = _space()
    assert space.names == PARAMETER_ORDER
    assert space.metadata["profile"] == "physics"
    expected = derive_physics_controller_initials(PROJECT_ROOT)
    evidence = space.metadata.get("current_pidf_feasibility_evidence")
    if evidence is not None:
        for name, value in evidence["selected_current_parameters"].items():
            expected[space.names.index(name)] = value
    assert np.allclose(
        space.initial,
        expected,
        rtol=1e-12,
        atol=0.0,
    )
    assert all(
        spec.source_kind in {
            "performance_target_seed_derived",
            "non_training_feasibility_candidate_derived",
            "approved_dobc_structure",
        }
        for spec in space.specs
    )


def test_parameter_metadata_links_the_model_and_six_metric_targets():
    config = load_physics_motor_config(PROJECT_ROOT)
    targets = load_controller_performance_targets(PROJECT_ROOT)
    space = _space()
    assert space.metadata["schema_version"] == 3
    assert space.metadata["physics_model"]["model_id"] == config.payload["model_id"]
    assert space.metadata["physics_model"]["primary_training_plant"]
    assert not space.metadata["physics_model"]["measured_frf_used_as_training_plant"]
    assert space.metadata["performance_targets"]["closed_loop_bandwidth_hz"] == {
        loop: targets.loop(loop).bandwidth_hz
        for loop in ("current", "speed", "position")
    }
    targets_path = PROJECT_ROOT / "config" / "controller_performance_targets.json"
    assert space.metadata["performance_targets"]["config_sha256"] == (
        hashlib.sha256(targets_path.read_bytes()).hexdigest()
    )
    assert space.metadata["performance_targets"][
        "bandwidth_is_not_relabelled_as_open_loop_crossover"
    ]
    for spec in space.specs:
        loop = "speed" if spec.module == "DOBC" else spec.module
        assert spec.sample_period_s == config.sample_period_s_for(loop)
    assert space.metadata["digital_conversion"]["sample_periods_s"] == (
        config.sample_periods_s
    )
    stages = {spec.name: spec.training_stage for spec in space.specs}
    assert stages["kgspeed"] == stages["tauspeed"] == 2


def test_normalized_mapping_round_trip_and_bounded_action():
    space = _space()
    normalized_initial = space.normalize(space.initial)
    assert np.all((-1.0 <= normalized_initial) & (normalized_initial <= 1.0))
    assert np.allclose(space.denormalize(normalized_initial), space.initial)

    grid = np.stack([space.lower, space.initial, space.upper])
    assert np.allclose(space.denormalize(space.normalize(grid)), grid)
    updated = space.apply_action(space.initial, np.ones(11))
    assert np.all(updated >= space.lower)
    assert np.all(updated <= space.upper)


def test_roundoff_sized_boundary_excursions_are_clipped_safely():
    space = _space()
    just_above_upper = np.nextafter(space.upper, np.inf)
    just_below_lower = np.nextafter(space.lower, -np.inf)

    upper_normalized = space.normalize(just_above_upper)
    lower_normalized = space.normalize(just_below_lower)

    assert np.all(upper_normalized == 1.0)
    assert np.all(lower_normalized == -1.0)
    assert np.all(space.denormalize(upper_normalized) <= space.upper)
    assert np.all(space.denormalize(lower_normalized) >= space.lower)


def test_repeated_actions_at_parameter_limits_remain_valid():
    space = _space()
    parameters = space.upper.copy()
    for _ in range(100):
        parameters = space.apply_action(parameters, np.ones(len(space.specs)))
    assert np.all(parameters == space.upper)


def test_material_boundary_violations_and_nonfinite_values_are_rejected():
    space = _space()
    invalid = space.upper.copy()
    kispeed_index = space.names.index("kispeed")
    invalid[kispeed_index] += max(1e-8, abs(space.upper[kispeed_index]) * 1e-8)
    with pytest.raises(ValueError, match="kispeed"):
        space.normalize(invalid)

    nonfinite = space.initial.copy()
    nonfinite[kispeed_index] = np.nan
    with pytest.raises(ValueError, match="finite"):
        space.normalize(nonfinite)


def test_all_eleven_parameters_remain_simulation_only():
    space = _space()
    assert len(space.specs) == 11
    assert all(spec.original_value is None for spec in space.specs)
    assert all(spec.hardware_status.startswith("simulation_only") for spec in space.specs)
    assert space.metadata["safety_policy"]["direct_hardware_use_allowed"] is False


def test_current_feasibility_evidence_excludes_validation_from_selection(tmp_path):
    config = load_physics_motor_config(PROJECT_ROOT)
    selection_groups = ["nominal", "corners_8", "dense_grid_125", "training_40"]
    report = {
        "run_kind": "non_training_multirate_current_pidf_feasibility_scan",
        "formal_configuration_modified": False,
        "formal_parameter_space_modified": False,
        "sac_training_executed": False,
        "sample_rates": [
            {
                "sample_rate_hz": 1.0 / config.sample_period_s_for("current"),
                "search": {
                    "candidate_selection_groups": selection_groups,
                    "validation_used_for_candidate_selection": False,
                },
                "selected": {
                    "current_parameters": {
                        "kpcurr": 10.0,
                        "kicurr": 5000.0,
                        "kdcurr": 0.002,
                    },
                    "groups": {
                        **{name: {"all_six_pass": True} for name in selection_groups},
                        "validation_16": {"all_six_pass": False},
                    },
                },
            }
        ],
    }
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    continuous_report = {
        "run_kind": "non_training_continuous_current_uncertainty_refinement",
        "formal_configuration_modified": False,
        "formal_parameter_space_modified": False,
        "sac_training_executed": False,
        "refinements": [
            {
                "sample_rate_hz": 1.0
                / config.sample_period_s_for("current"),
                "current_parameters": report["sample_rates"][0]["selected"][
                    "current_parameters"
                ],
                "continuous_box_all_six_pass": True,
            }
        ],
    }
    (tmp_path / "continuous_worst_case_report.json").write_text(
        json.dumps(continuous_report),
        encoding="utf-8",
    )
    evidence = load_current_pidf_feasibility_evidence(PROJECT_ROOT, path)
    assert evidence["validation_used_for_candidate_selection"] is False
    assert evidence["selected_current_parameters"]["kdcurr"] == 0.002

    report["sample_rates"][0]["search"][
        "validation_used_for_candidate_selection"
    ] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        load_current_pidf_feasibility_evidence(PROJECT_ROOT, path)
