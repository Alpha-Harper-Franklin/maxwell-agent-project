from __future__ import annotations

import re

from .models import ElectromagnetDesign, RequirementCheck, RequirementEvaluation


COPPER_RESISTIVITY_OHM_M = 1.724e-8
COPPER_FILL_FACTOR = 0.6


def build_requirement_evaluation(
    design: ElectromagnetDesign,
    outputs: dict[str, float | str] | None,
    run_status: str,
) -> RequirementEvaluation:
    outputs = outputs or {}
    checks: list[RequirementCheck] = []
    requirement = design.source_requirement or design.summary or ""
    constraints = _extract_numeric_constraints(requirement)
    electrical_estimate = _estimate_electrical_behavior(design)

    if electrical_estimate["coil_resistance_ohm"] is not None:
        outputs.setdefault("estimated_coil_resistance_ohm", electrical_estimate["coil_resistance_ohm"])
    if electrical_estimate["supply_current_a"] is not None:
        outputs.setdefault("estimated_current_at_supply_a", electrical_estimate["supply_current_a"])

    if run_status == "completed":
        checks.append(
            RequirementCheck(
                name="仿真执行",
                status="passed",
                detail="Maxwell 已完成建模、求解和结果导出。",
            )
        )
    elif run_status == "blocked":
        checks.append(
            RequirementCheck(
                name="仿真执行",
                status="failed",
                detail="本次没有完成真实 Maxwell 仿真，当前只有规划或结构化结果。",
            )
        )
    else:
        checks.append(
            RequirementCheck(
                name="仿真执行",
                status="failed",
                detail="Maxwell 执行失败，本次没有形成可信的仿真结果。",
            )
        )

    if constraints["air_gap_mm"] is not None:
        air_gap_ok = abs(design.air_gap_mm - constraints["air_gap_mm"]) <= 1e-6
        checks.append(
            RequirementCheck(
                name="气隙约束",
                status="passed" if air_gap_ok and run_status == "completed" else "failed" if not air_gap_ok else "unverified",
                detail=(
                    f"需求要求气隙 {constraints['air_gap_mm']:g} mm，本次模型按 {design.air_gap_mm:g} mm 执行。"
                    if air_gap_ok
                    else f"需求要求气隙 {constraints['air_gap_mm']:g} mm，但本次模型按 {design.air_gap_mm:g} mm 执行。"
                ),
            )
        )

    current_min_a = constraints["current_min_a"]
    current_limit_a = constraints["current_limit_a"]
    current_target_a = constraints["current_target_a"]

    if current_min_a is not None and current_limit_a is not None:
        design_current_ok = current_min_a - 1e-9 <= design.current_a <= current_limit_a + 1e-9
        checks.append(
            RequirementCheck(
                name="目标电流范围",
                status="passed" if design_current_ok else "failed",
                detail=(
                    f"需求要求建模电流落在 {current_min_a:g}-{current_limit_a:g} A，本次模型按 {design.current_a:g} A 执行。"
                    if design_current_ok
                    else f"需求要求建模电流落在 {current_min_a:g}-{current_limit_a:g} A，但本次模型按 {design.current_a:g} A 执行。"
                ),
            )
        )
    else:
        if current_limit_a is not None:
            current_ok = design.current_a <= current_limit_a + 1e-9
            checks.append(
                RequirementCheck(
                    name="电流上限",
                    status="passed" if current_ok else "failed",
                    detail=(
                        f"模型执行电流为 {design.current_a:g} A，满足不超过 {current_limit_a:g} A。"
                        if current_ok
                        else f"模型执行电流为 {design.current_a:g} A，已超过 {current_limit_a:g} A。"
                    ),
                )
            )

        if current_target_a is not None:
            exact_current_ok = abs(design.current_a - current_target_a) <= 1e-6
            checks.append(
                RequirementCheck(
                    name="目标电流",
                    status="passed" if exact_current_ok else "failed",
                    detail=(
                        f"需求指定电流 {current_target_a:g} A，本次模型按 {design.current_a:g} A 执行。"
                        if exact_current_ok
                        else f"需求指定电流 {current_target_a:g} A，但本次模型按 {design.current_a:g} A 执行。"
                    ),
                )
            )

    if constraints["supply_voltage_v"] is not None:
        supply_current = electrical_estimate["supply_current_a"]
        coil_resistance = electrical_estimate["coil_resistance_ohm"]
        if supply_current is None or coil_resistance is None:
            checks.append(
                RequirementCheck(
                    name="供电电压条件",
                    status="unverified",
                    detail=f"需求给出 {constraints['supply_voltage_v']:g} V 供电，但当前无法估算线圈电阻和该电压下的工作电流。",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="供电电压条件",
                    status="passed",
                    detail=(
                        f"按当前几何估算，线圈电阻约 {coil_resistance:.3g} Ω，"
                        f"在 {constraints['supply_voltage_v']:g} V 下电流约 {supply_current:.3g} A。"
                    ),
                )
            )
            if current_min_a is not None and current_limit_a is not None:
                voltage_current_ok = current_min_a - 1e-9 <= supply_current <= current_limit_a + 1e-9
                checks.append(
                    RequirementCheck(
                        name="电压/电流联合约束",
                        status="passed" if voltage_current_ok else "failed",
                        detail=(
                            f"在 {constraints['supply_voltage_v']:g} V 下，估算电流约 {supply_current:.3g} A，满足 {current_min_a:g}-{current_limit_a:g} A 区间。"
                            if voltage_current_ok
                            else f"在 {constraints['supply_voltage_v']:g} V 下，估算电流约 {supply_current:.3g} A，不满足 {current_min_a:g}-{current_limit_a:g} A 区间。"
                        ),
                    )
                )
            elif current_limit_a is not None:
                voltage_current_ok = supply_current <= current_limit_a + 1e-9
                checks.append(
                    RequirementCheck(
                        name="电压/电流联合约束",
                        status="passed" if voltage_current_ok else "failed",
                        detail=(
                            f"在 {constraints['supply_voltage_v']:g} V 下，估算电流约 {supply_current:.3g} A，满足不超过 {current_limit_a:g} A。"
                            if voltage_current_ok
                            else f"在 {constraints['supply_voltage_v']:g} V 下，估算电流约 {supply_current:.3g} A，不满足不超过 {current_limit_a:g} A。"
                        ),
                    )
                )

    if constraints["target_force_n"] is not None:
        if "force_n" in outputs:
            force_value = float(outputs["force_n"])
            target_ok = force_value >= constraints["target_force_n"]
            checks.append(
                RequirementCheck(
                    name="目标吸力",
                    status="passed" if target_ok else "failed",
                    detail=(
                        f"计算吸力约 {force_value:.4g} N，达到目标 {constraints['target_force_n']:g} N。"
                        if target_ok
                        else f"计算吸力约 {force_value:.4g} N，未达到目标 {constraints['target_force_n']:g} N。"
                    ),
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="目标吸力",
                    status="unverified",
                    detail=f"需求给出了目标吸力 {constraints['target_force_n']:g} N，但当前版本还未提取吸力结果。",
                )
            )

    if "max_flux_density_t" in outputs:
        checks.append(
            RequirementCheck(
                name="磁密结果",
                status="passed",
                detail=f"本次已提取最大磁密，约为 {float(outputs['max_flux_density_t']):.4g} T。",
            )
        )

    overall_status = _summarize_overall_status(checks)
    summary = _build_summary(overall_status, run_status)
    return RequirementEvaluation(overall_status=overall_status, summary=summary, checks=checks)


def _extract_numeric_constraints(requirement: str) -> dict[str, float | None]:
    current_min_a, current_max_a = _extract_current_range(requirement)
    return {
        "supply_voltage_v": _first_number(requirement, r"([0-9]+(?:\.[0-9]+)?)\s*V"),
        "air_gap_mm": _first_number(requirement, r"气隙\s*([0-9]+(?:\.[0-9]+)?)\s*mm"),
        "current_min_a": current_min_a,
        "current_limit_a": _first_number(
            requirement,
            r"(?:电流|current).{0,8}?(?:不超过|不大于|上限|最大|max)\s*([0-9]+(?:\.[0-9]+)?)\s*A",
        ) or current_max_a,
        "current_target_a": _first_number(
            requirement,
            r"(?:电流|current)\s*(?:为|取值为|设为|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*A",
        ) if current_max_a is None else None,
        "target_force_n": _first_number(
            requirement,
            r"(?:吸力|力).{0,8}?(?:至少|不低于|不小于|达到|大于等于)\s*([0-9]+(?:\.[0-9]+)?)\s*N",
        ),
    }


def _first_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _extract_current_range(text: str) -> tuple[float | None, float | None]:
    match = re.search(
        r"(?:电流|current)\s*([0-9]+(?:\.[0-9]+)?)\s*A\s*(?:-|~|～|到|至)\s*([0-9]+(?:\.[0-9]+)?)\s*A?",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None, None
    low = float(match.group(1))
    high = float(match.group(2))
    if low > high:
        low, high = high, low
    return low, high


def _estimate_electrical_behavior(design: ElectromagnetDesign) -> dict[str, float | None]:
    if design.coil_turns <= 0:
        return {"coil_resistance_ohm": None, "supply_current_a": None}

    window_area_mm2 = design.coil_width_mm * design.coil_height_mm
    if window_area_mm2 <= 0:
        return {"coil_resistance_ohm": None, "supply_current_a": None}

    conductor_area_m2 = (window_area_mm2 * COPPER_FILL_FACTOR / design.coil_turns) * 1e-6
    if conductor_area_m2 <= 0:
        return {"coil_resistance_ohm": None, "supply_current_a": None}

    mean_turn_length_mm = 2.0 * (design.coil_height_mm + design.coil_width_mm + design.core_thickness_mm)
    total_length_m = design.coil_turns * mean_turn_length_mm * 1e-3
    coil_resistance = COPPER_RESISTIVITY_OHM_M * total_length_m / conductor_area_m2

    supply_current = None
    if design.supply_voltage_v is not None and coil_resistance > 0:
        supply_current = design.supply_voltage_v / coil_resistance

    return {
        "coil_resistance_ohm": coil_resistance,
        "supply_current_a": supply_current,
    }


def _summarize_overall_status(checks: list[RequirementCheck]) -> str:
    statuses = {item.status for item in checks}
    if "failed" in statuses:
        return "failed"
    if "unverified" in statuses and "passed" in statuses:
        return "partial"
    if statuses == {"passed"}:
        return "passed"
    return "unverified"


def _build_summary(overall_status: str, run_status: str) -> str:
    if run_status != "completed":
        return "当前没有完成一次可信的 Maxwell 仿真，因此无法判断用户需求是否满足。"
    if overall_status == "passed":
        return "本次仿真已完成，且当前已验证的需求项均满足。"
    if overall_status == "failed":
        return "本次仿真虽然执行完成，但存在未满足的需求项。"
    if overall_status == "partial":
        return "本次仿真已完成，但目前只验证了部分需求，仍有关键项未验证。"
    return "本次仿真已完成，但当前版本还不能对需求满足性给出明确结论。"
