import ast
import os
import re


EXPECTED_SIGNATURES = {
    "count_steps.py": {
        "count_steps": {
            "args": ["model_name: str", "instance_id: str"],
            "return": "int",
        },
        "write_to_json": {
            "args": [],
            "return": None,
        },
        "plot_violin": {
            "args": [],
            "return": None,
        },
    },
    "thought_analysis.py": {
        "find_relevance": {
            "args": ["model_name: str", "instance_id: str"],
            "return": "list[str]",
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
                expected_signature = f"{function_name}({', '.join(expected_args)})"
                actual_arg_text = "unsupported parameter layout" if actual_args is None else ", ".join(actual_args)
                actual_signature = f"{function_name}({actual_arg_text})"
                if expected_return is not None:
                    expected_signature += f" -> {expected_return}"
                if actual_return is not None:
                    actual_signature += f" -> {actual_return}"

                print(f"  [MISMATCH] {filename}")
                print(f"    expected: {expected_signature}")
                print(f"    found:    {actual_signature}")
                all_valid = False
            else:
                print(f"  [FOUND] {filename}: {function_name}")

    return all_valid

## execute this script under the milestone1 directory to check repo structure
def check_repo_structure():

    required_files = [
        "count_steps.py",
        "number_of_steps.json",
        "number_of_steps.jpeg",
        "issue_entities.json",
        "thought_analysis.py",
        "thought_entity_relevance.json",
    ]

    required_dirs = [
        "inspector_txt",
        "../Trajectories",
        "../Trajectories/gpt-5-mini",
        "../Trajectories/deepseek-v3",
    ]

    all_exist = True

    print("Checking required directories:")
    for dirname in required_dirs:
        exists = os.path.isdir(dirname)
        status = "FOUND" if exists else "MISSING"
        print(f"  [{status}] {dirname}")
        if not exists:
            all_exist = False

    print("\nChecking required files:")
    for filename in required_files:
        filepath = os.path.join(".", filename)
        exists = os.path.isfile(filepath)
        status = "FOUND" if exists else "MISSING"
        print(f"  [{status}] {filename}")
        if not exists:
            all_exist = False

    print("\nChecking trajectory files:")

    traj_dirs = [
        "../Trajectories/gpt-5-mini",
        "../Trajectories/deepseek-v3",
    ]

    for traj_dir in traj_dirs:
        if not os.path.isdir(traj_dir):
            print(f"  [MISSING DIR] {traj_dir}")
            all_exist = False
            continue

        files = os.listdir(traj_dir)
        traj_files = [f for f in files if f.endswith(".traj")]

        if traj_files:
            print(f"  [FOUND] {len(traj_files)} .traj files in {traj_dir}")
        else:
            print(f"  [MISSING] No .traj files in {traj_dir}")
            all_exist = False

    print("\nChecking inspector_txt files:")

    inspector_dir = "inspector_txt"

    if os.path.isdir(inspector_dir):
        files = os.listdir(inspector_dir)

        gpt_files = [f for f in files if re.match(r"gpt-5-mini-.+\.txt$", f)]
        deepseek_files = [f for f in files if re.match(r"deepseek-v3-.+\.txt$", f)]

        if gpt_files:
            print(f"  [FOUND] {len(gpt_files)} gpt-5-mini inspector files")
        else:
            print("  [MISSING] gpt-5-mini-<instance_id>.txt files")
            all_exist = False

        if deepseek_files:
            print(f"  [FOUND] {len(deepseek_files)} deepseek-v3 inspector files")
        else:
            print("  [MISSING] deepseek-v3-<instance_id>.txt files")
            all_exist = False
    else:
        print("  [MISSING DIR] inspector_txt")
        all_exist = False

    return all_exist


if __name__ == "__main__":
    structure_ok = check_repo_structure()
    signatures_ok = check_method_signatures()
    result = structure_ok and signatures_ok
    if result:
        print("\nAll required files, folders, and method signatures are valid.")
    else:
        print("\nSome required files, folders, or method signatures are invalid.")
