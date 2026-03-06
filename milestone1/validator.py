import os
import re

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
    result = check_repo_structure()
    if result:
        print("\nAll required files and folders are present.")
    else:
        print("\nSome required files or folders are missing.")