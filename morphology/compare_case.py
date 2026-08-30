import json
import argparse
from pathlib import Path
import math


FEATURES = [
    "nuclei_total",
    "nuclei_mean_area_px2",
    "nuclei_mean_perimeter_px",
    "nuclei_mean_eccentricity",
    "nuclei_mean_circularity",
    "glands_total",
    "glands_mean_area_px2",
    "glands_mean_perimeter_px",
    "glands_mean_width_px",
    "glands_mean_height_px",
    "glands_mean_aspect_ratio",
    "glands_mean_circularity",
]


def normalized_distance(current, reference):
    values = []

    for feature in FEATURES:
        x = float(current.get(feature, 0))
        y = float(reference.get(feature, 0))

        denominator = max(abs(x), abs(y), 1.0)

        difference = abs(x - y) / denominator

        values.append(difference)

    return sum(values) / len(values)


def similarity_score(distance):
    return max(0.0, 1.0 - distance) * 100.0


def main():

    parser = argparse.ArgumentParser(
        description="Compare colon morphology cases"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        current_case = json.load(f)

    reference_dir = Path(args.references)

    results = []

    for reference_file in reference_dir.rglob("*.json"):

        with open(reference_file, "r", encoding="utf-8") as f:
            reference = json.load(f)

        distance = normalized_distance(
            current_case,
            reference
        )

        score = similarity_score(distance)

        results.append({
            "reference": reference_file.stem,
            "class": reference.get("class", "unknown"),
            "normalized_distance": distance,
            "similarity_percent": score
        })

    results.sort(
        key=lambda x: x["similarity_percent"],
        reverse=True
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_data = {
        "case_id": current_case["case_id"],
        "comparisons": results
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            result_data,
            f,
            indent=2
        )

    print("=" * 60)
    print("COLON MORPHOLOGY REFERENCE COMPARISON")
    print("=" * 60)

    if not results:

        print("No reference cases found.")

    else:

        for result in results:

            print(
                f'{result["reference"]} | '
                f'{result["class"]} | '
                f'Similarity: '
                f'{result["similarity_percent"]:.2f}%'
            )

        print()
        print("TOP MATCH")
        print("-" * 30)

        best = results[0]

        print("Reference:", best["reference"])
        print("Class:", best["class"])
        print(
            "Similarity:",
            f'{best["similarity_percent"]:.2f}%'
        )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()