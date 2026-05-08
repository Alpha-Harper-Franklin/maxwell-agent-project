from __future__ import annotations


class RequirementPlanningError(Exception):
    def __init__(
        self,
        message: str,
        reason_code: str = "planning_error",
        intake: object | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.reason_code = reason_code
        self.intake = intake


class UnsupportedRequirementError(RequirementPlanningError):
    def __init__(self, message: str, intake: object | None = None) -> None:
        super().__init__(message=message, reason_code="unsupported_requirement", intake=intake)


class UnknownPrimitiveError(RequirementPlanningError):
    def __init__(
        self,
        message: str,
        primitive_token: str,
        raw_object: dict[str, object] | None = None,
        intake: object | None = None,
    ) -> None:
        super().__init__(message=message, reason_code="unknown_primitive", intake=intake)
        self.primitive_token = primitive_token
        self.raw_object = raw_object or {}
