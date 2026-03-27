import os
import ast
## execute this script under the milestone4 directory to check if all required files are present

EXPECTED_SIGNATURES = {
    "milestone4.py": {
        "extract_phase_sequences": {
            "args": ["model_name: str", "instance_id: str"],
            "return": "list",
        },
        "shortcuts_backtracks_detection": {
            "args": ["model_name: str", "instance_id: str"],
            "return": "dict",
        },
        "check_plan_compliance": {
            "args": ["model_name: str", "instance_id: str"],
            "return": "dict",
        },
        "mine_shared_strategy": {
            "args": ["model_name: str"],
            "return": "list",
        },
    },
}


def _format_arg(arg_node):
    annotation = ""
    if arg_node.annotation is not None:
        annotation = f": {ast.unparse(arg_node.annotation)}"
    return f"{arg_node.arg}{annotation}"


def _extract_signatures(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            source = file.read()
    except OSError as exc:
        return None, f"Could not read file: {exc}"

    try:
        module = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        location = f"line {exc.lineno}"
        if exc.offset is not None:
            location += f", column {exc.offset}"
        return None, f"Invalid Python syntax at {location}: {exc.msg}"

    signatures = {}
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            if node.args.posonlyargs or node.args.vararg or node.args.kwonlyargs or node.args.kwarg:
                args = None
            else:
                args = [_format_arg(arg) for arg in node.args.args]

            signatures[node.name] = {
                "args": args,
                "return": ast.unparse(node.returns) if node.returns is not None else None,
            }

    return signatures, None


def check_method_signatures():
    all_valid = True
    print("\nChecking required method signatures:")

    for filename, expected_functions in EXPECTED_SIGNATURES.items():
        filepath = os.path.join(".", filename)
        if not os.path.isfile(filepath):
            print(f"  [SKIPPED] {filename} (file missing)")
            all_valid = False
            continue

        signatures, error = _extract_signatures(filepath)
        if error is not None:
            print(f"  [INVALID] {filename}: {error}")
            all_valid = False
            continue

        for function_name, expected in expected_functions.items():
            actual = signatures.get(function_name)
            if actual is None:
                print(f"  [MISSING] {filename}: function `{function_name}`")
                all_valid = False
                continue

            actual_args = actual["args"]
            expected_args = expected["args"]
            actual_return = actual["return"]
            expected_return = expected["return"]

            if actual_args != expected_args or actual_return != expected_return:
                expected_signature = f"{function_name}({', '.join(expected_args)}) -> {expected_return}"
                actual_arg_text = "unsupported parameter layout" if actual_args is None else ", ".join(actual_args)
                actual_signature = f"{function_name}({actual_arg_text})"
                if actual_return is not None:
                    actual_signature += f" -> {actual_return}"

                print(f"  [MISMATCH] {filename}")
                print(f"    expected: {expected_signature}")
                print(f"    found:    {actual_signature}")
                all_valid = False
            else:
                print(f"  [FOUND] {filename}: {function_name}")

    return all_valid



def check_files_exist():
    required_files = [
        "milestone4.py",
        "phase_sequence.json",
        "shortcuts_backtracks.json",
        "plan_compliance.json",
        "shared_strategy.json"
    ]

    all_exist = True
    for filename in required_files:
        filepath = os.path.join(".", filename)
        exists = os.path.isfile(filepath)
        status = "FOUND" if exists else "MISSING"
        print(f"  [{status}] {filename}")
        if not exists:
            all_exist = False

    return all_exist


if __name__ == "__main__":
    result = check_files_exist()
    if result:
        print("\nAll required files are present.")
    else:
        print("\nSome required files are missing.")
        
    signatures_ok = check_method_signatures()
    if signatures_ok:
        print("\nAll required method signatures are valid.")
    else:
        print("\nSome required method signatures are invalid.")
