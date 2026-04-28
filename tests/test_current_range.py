from maxwell_agent.evaluation import build_requirement_evaluation
from maxwell_agent.llm_client import _normalize_design_payload
from maxwell_agent.models import ElectromagnetDesign


def test_normalize_design_payload_extracts_current_range() -> None:
    payload = _normalize_design_payload(
        {"summary": "demo"},
        "做一个24V直流电磁铁，气隙2mm，电流2A-4A，尽量提高吸力，外形不要太大",
    )

    assert payload["current_min_a"] == 2.0
    assert payload["current_limit_a"] == 4.0


def test_requirement_evaluation_fails_joint_voltage_current_range_when_out_of_range() -> None:
    design = ElectromagnetDesign(
        source_requirement="做一个24V直流电磁铁，气隙2mm，电流2A-4A，尽量提高吸力，外形不要太大",
        summary="demo",
        objective="maximize_force",
        supply_voltage_v=24.0,
        current_min_a=2.0,
        current_limit_a=4.0,
        current_a=4.0,
        air_gap_mm=2.0,
        coil_turns=400,
        coil_width_mm=12.0,
        coil_height_mm=20.0,
        core_thickness_mm=10.0,
    )

    evaluation = build_requirement_evaluation(
        design,
        outputs={"max_flux_density_t": 0.01},
        run_status="completed",
    )

    assert evaluation.overall_status == "failed"
    assert any(item.name == "电压/电流联合约束" and item.status == "failed" for item in evaluation.checks)
