#!/usr/bin/env python3
"""Count SpaceX launches per rocket using the unofficial API.

Output lines are sorted by number of launches (descending) and then by
rocket name (A–Z), formatted as:
<Rocket Name>: <count>
"""

import requests


def main():
    launches_resp = requests.get("https://api.spacexdata.com/v4/launches")
    launches = launches_resp.json()

    # Count launches per rocket id
    counts = {}
    for launch in launches:
        rid = launch.get("rocket")
        if not rid:
            continue
        counts[rid] = counts.get(rid, 0) + 1

    # Fetch all rockets once to map id -> name
    rockets_resp = requests.get("https://api.spacexdata.com/v4/rockets")
    rockets = rockets_resp.json()
    id_to_name = {r.get("id"): r.get("name") for r in rockets}

    # Build (name, count) entries for rockets that appear in launches
    entries = []
    for rid, cnt in counts.items():
        name = id_to_name.get(rid, "")
        entries.append((name, cnt))

    # Sort by count desc, then name asc
    entries.sort(key=lambda x: (-x[1], x[0]))

    for name, cnt in entries:
        print(f"{name}: {cnt}")


if __name__ == '__main__':
    main()
