import os
## execute this script under the milestone2 directory to check if all required files are present
def check_files_exist():
    required_files = [
        "milestone2.py",
        "locate_generated_tests.json",
        "fail_to_pass.json",
        "locate_navigation.json",
        "count_tool_use.json",
        "fail_to_pass.jpeg",
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
