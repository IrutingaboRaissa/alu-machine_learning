#!/usr/bin/env python3
"""Display the soonest upcoming SpaceX launch using the unofficial API.

Format:
<launch name> (<date_local>) <rocket name> - <launchpad name> (<locality>)
"""

import requests


def main():
    launches_url = "https://api.spacexdata.com/v4/launches/upcoming"
    resp = requests.get(launches_url)
    launches = resp.json()

    # Pick the soonest upcoming launch by 'date_unix'.
    # 'sorted' is stable, so equal dates keep original API order.
    launches_sorted = sorted(launches, key=lambda x: int(x.get("date_unix", 0)))
    launch = launches_sorted[0]

    launch_name = launch["name"]
    date_local = launch["date_local"]
    rocket_id = launch["rocket"]
    launchpad_id = launch["launchpad"]

    rocket_name = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    ).json()["name"]

    launchpad = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    ).json()
    launchpad_name = launchpad["name"]
    launchpad_locality = launchpad["locality"]

    print(
        f"{launch_name} ({date_local}) {rocket_name} - {launchpad_name} ({launchpad_locality})"
    )


if __name__ == '__main__':
    main()
