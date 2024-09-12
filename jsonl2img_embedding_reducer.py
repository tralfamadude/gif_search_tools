#!/usr/bin/env python3
import sys
import json
from collections import defaultdict

def main():
    vector_count_by_hash = defaultdict(int)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        hash_value = data.get("hash")
        if hash_value:
            vector_count_by_hash[hash_value] += 1

    # Calculate histogram
    counts = list(vector_count_by_hash.values())
    max_count = max(counts) if counts else 1
    histogram = [counts.count(i) / len(counts) for i in range(1, max_count + 1)]

    # Print histogram in Python list syntax
    print(histogram)

if __name__ == "__main__":
    main()

