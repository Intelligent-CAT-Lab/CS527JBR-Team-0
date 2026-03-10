import csv
import io
import requests
import os
import sys

SPREADSHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1_ty1HJ6NEdSfo6D8rqU5i7-griR02PRjdAlfA8pWRiQ/export?format=csv&gid=0"
)


def read_spreadsheet() -> dict[str, list[str]]:
    """Read the Google Sheets spreadsheet and return a dictionary mapping
    group_id to a list of '{model}-{instance}' strings."""
    response = requests.get(SPREADSHEET_URL)
    response.raise_for_status()

    reader = csv.DictReader(io.StringIO(response.text))
    result: dict[str, list[str]] = {}
    for row in reader:
        group_id = int(row["group_id"])
        entry = f"{row['model']}-{row['instance']}"
        result.setdefault(group_id, []).append(entry)
    return result

def collect_assigned_trajectories(group_id):
    file_names = []
    assigned_trajs = read_spreadsheet().get(int(group_id), [])
    for at in assigned_trajs:
        file_names.append(f"./graphs/{at}.json")
        file_names.append(f"./graphs/{at}.pdf")
    return file_names


def check_files_exist(group_id):
    traj_files = collect_assigned_trajectories(group_id)
    required_files = traj_files + [
        "milestone3.py",
        "graph_metrics.json"
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
    ## usage: python validator.py <group_id> e.g., python validator.py 1
    group_id  = sys.argv[1] if len(sys.argv) > 1 else None
    result = check_files_exist(group_id)
    if result:
        print("\nAll required files are present.")
    else:
        print("\nSome required files are missing.")