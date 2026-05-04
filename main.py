"""
  python main.py           
  python main.py --part 1  
  python main.py --part 2 
  python main.py --part 3 
"""

import argparse
import sys

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       Heart Failure Clinical Records — ML Project           ║
║       Dataset: UCI / Kaggle (299 unique records)            ║
╠══════════════════════════════════════════════════════════════╣
║  Part 1 — EDA & Pre-processing                              ║
║  Part 2 — Unsupervised ML  (Hierarchical + K-Means)         ║
║  Part 3 — Supervised ML    (ANN + LogReg + RandomForest)    ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(description="Heart Failure ML Project")
    parser.add_argument("--part", type=int, choices=[1, 2, 3],
                        help="Run only a specific part (1, 2, or 3). "
                             "Omit to run all parts sequentially.")
    args = parser.parse_args()

    print(BANNER)

    parts_to_run = [args.part] if args.part else [1, 2, 3]

    if 1 in parts_to_run:
        print("\n" + "─"*62)
        print("  PART 1 — EDA & DATA PRE-PROCESSING")
        print("─"*62)
        from part1_eda import run as run_part1
        run_part1()

    if 2 in parts_to_run:
        print("\n" + "─"*62)
        print("  PART 2 — UNSUPERVISED MACHINE LEARNING")
        print("─"*62)
        from part2_clustering import run as run_part2
        run_part2()

    if 3 in parts_to_run:
        print("\n" + "─"*62)
        print("  PART 3 — SUPERVISED MACHINE LEARNING")
        print("─"*62)
        from part3_classification import run as run_part3
        run_part3()


if __name__ == "__main__":
    main()
