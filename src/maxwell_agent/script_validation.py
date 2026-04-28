from __future__ import annotations

import ast

from .models import GeneratedMaxwellScript, ScriptStaticCheck


ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "json",
    "math",
    "typing",
    "pathlib",
    "ansys",
}

BANNED_IMPORT_ROOTS = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "shutil",
    "tempfile",
    "multiprocessing",
    "asyncio",
}

BANNED_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
}

BANNED_ATTR_NAMES = {
    "system",
    "popen",
    "run",
    "call",
    "check_call",
    "check_output",
    "write_text",
    "write_bytes",
    "unlink",
    "rmtree",
}


def static_check_generated_script(script: GeneratedMaxwellScript) -> ScriptStaticCheck:
    errors: list[str] = []
    warnings: list[str] = []
    imported_modules: list[str] = []

    try:
        tree = ast.parse(script.code)
    except SyntaxError as exc:
        return ScriptStaticCheck(
            passed=False,
            required_entrypoint=script.entrypoint,
            imported_modules=[],
            errors=[f"Python 语法错误: {exc.msg} (line {exc.lineno})"],
            warnings=[],
        )

    has_entrypoint = False
    has_return = False
    has_maxwell_ref = any(token in script.code for token in ("Maxwell2d", "Maxwell3d"))
    has_save_project = "save_project" in script.code

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                imported_modules.append(alias.name)
                if root in BANNED_IMPORT_ROOTS:
                    errors.append(f"不允许导入模块: {alias.name}")
                elif root not in ALLOWED_IMPORT_ROOTS:
                    warnings.append(f"发现未列入白名单的导入: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            imported_modules.append(module)
            if root in BANNED_IMPORT_ROOTS:
                errors.append(f"不允许导入模块: {module}")
            elif root and root not in ALLOWED_IMPORT_ROOTS:
                warnings.append(f"发现未列入白名单的导入: {module}")

        if isinstance(node, ast.FunctionDef) and node.name == script.entrypoint:
            has_entrypoint = True
            if len(node.args.args) != 1:
                errors.append(f"入口函数 {script.entrypoint} 必须接收一个参数 job。")
            has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                errors.append(f"禁止调用: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_ATTR_NAMES:
                errors.append(f"禁止调用危险方法: .{node.func.attr}()")

    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            continue
        errors.append(f"顶层语句不允许出现 {type(node).__name__}。")

    if not has_entrypoint:
        errors.append(f"缺少入口函数 {script.entrypoint}(job: dict) -> dict。")
    if has_entrypoint and not has_return:
        errors.append(f"入口函数 {script.entrypoint} 必须返回一个 dict。")
    if not has_maxwell_ref:
        warnings.append("脚本中未显式引用 Maxwell2d 或 Maxwell3d，可能无法真正建模。")
    if not has_save_project:
        warnings.append("脚本中未显式调用 save_project()。")

    return ScriptStaticCheck(
        passed=not errors,
        required_entrypoint=script.entrypoint,
        imported_modules=sorted({item for item in imported_modules if item}),
        errors=errors,
        warnings=warnings,
    )
