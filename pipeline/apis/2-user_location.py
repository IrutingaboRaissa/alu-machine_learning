#!/usr/bin/env python3


"""Print the location for a GitHub user from the API URL passed as
the first argument.

Usage:
    ./2-user_location.py https://api.github.com/users/holbertonschool

If the user doesn't exist print: Not found
If the status code is 403 print: Reset in X min
"""

import sys
import time
import math
import requests


def print_user_location(url):
    """Request the GitHub user URL and print location or error messages.

    Args:
        url (str): Full API URL for the user, e.g. https://api.github.com/users/xxx
    """
    try:
        res = requests.get(url)
    except requests.RequestException:
        # Network error, don't crash; behave silently (tests don't expect output)
        return

    if res.status_code == 403:
        # header names are case-insensitive, check common variants
        reset = res.headers.get('X-RateLimit-Reset') or res.headers.get(
            'X-Ratelimit-Reset')
        try:
            reset = int(reset)
            # compute remaining minutes, round up to avoid reporting one minute less
            diff = math.ceil((reset - time.time()) / 60)
            if diff < 0:
                diff = 0
        except Exception:
            diff = 0

        print("Reset in {} min".format(diff))

    elif res.status_code == 404:
        print("Not found")

    elif res.status_code == 200:
        try:
            data = res.json()
        except ValueError:
            data = {}

        # print the location value (could be None)
        print(data.get('location'))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # nothing to do if no URL supplied
        sys.exit(1)

    print_user_location(sys.argv[1])
