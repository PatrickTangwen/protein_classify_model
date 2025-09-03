import os
import glob
import json
import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd


def compute_family_id(subfamily_id: str) -> str:
    """
    Derive the family identifier from a subfamily identifier by
    taking the first three dot-separated parts.
    Example: X.Y.Z.W -> X.Y.Z
    """
    return ".".join(subfamily_id.split(".")[:3])


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Build a simple GitHub-flavored Markdown table.
    """
    header_line = " | ".join(headers)
    sep_line = " | ".join(["---"] * len(headers))
    row_lines = [" | ".join(map(str, r)) for r in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def _read_kfold_sizes(results_dir: str, level: str) -> Optional[Dict[str, List[int]]]:
    """
    Read one available kfold_metrics.json under results_dir/kfold/<level>/*/kfold_metrics.json
    and return per-fold train/val sizes if present.
    """
    pattern = os.path.join(results_dir, "kfold", level, "*", "kfold_metrics.json")
    matches = glob.glob(pattern)
    if not matches:
        return None
    try:
        with open(matches[0], "r", encoding="utf-8") as f:
            payload = json.load(f)
        per_fold = payload.get("per_fold", [])
        num_train = [int(x.get("num_train")) for x in per_fold if "num_train" in x]
        num_val = [int(x.get("num_val")) for x in per_fold if "num_val" in x]
        if num_train and num_val and len(num_train) == len(num_val):
            return {"num_train": num_train, "num_val": num_val}
    except Exception:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset statistics from data_new.csv")
    parser.add_argument(
        "--level",
        choices=["subfamily", "family", "both"],
        default="both",
        help="Classification level for stats. 'both' prints family and subfamily.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to data_new.csv. Defaults to benchmark_scripts.config.PROTEIN_DATA_PATH",
    )
    parser.add_argument(
        "--save-markdown",
        action="store_true",
        help="Save a Markdown summary table under results directory.",
    )
    parser.add_argument(
        "--include-kfold",
        action="store_true",
        help="Include k-fold train/val sizes in the summary if found.",
    )
    args = parser.parse_args()

    # Resolve data path from config if not provided
    if args.data_path is None:
        try:
            from benchmark_scripts.config import PROTEIN_DATA_PATH as DEFAULT_DATA_PATH
        except Exception:
            # Fallback to repo default location
            repo_root = os.path.dirname(os.path.abspath(__file__))
            DEFAULT_DATA_PATH = os.path.join(os.path.dirname(repo_root), "data_source", "data_new.csv")
        data_path = DEFAULT_DATA_PATH
    else:
        data_path = args.data_path

    # Load utilities
    from data_loader import load_protein_data, load_superfamily_map
    from data_splitting import custom_split_dataset
    from data_splitting_same_sup import custom_split_dataset_with_negatives
    try:
        from benchmark_scripts.config_same_sup import RESULTS_DIR as SAME_SUP_RESULTS
    except Exception:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        SAME_SUP_RESULTS = os.path.join(os.path.dirname(repo_root), "benchmark_results_same_sup")

    def compute_for_level(level: str) -> Dict[str, any]:
        df_level = load_protein_data(data_path, level=level)
        if "Family" not in df_level.columns:
            df_level["Family"] = df_level["Subfamily"].apply(compute_family_id)

        num_proteins = len(df_level)
        num_subfamilies = df_level["Subfamily"].nunique()
        num_families = df_level["Family"].nunique()

        # basic split
        train_idx, test_idx = custom_split_dataset(df_level, level=level)
        basic_train = len(train_idx)
        basic_test = len(test_idx)

        # same-superfamily negative-control validation
        super_map = load_superfamily_map(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_source", "fam2supefamily.csv"))
        train2, val_with_negs, is_neg_map, target_test_mapping = custom_split_dataset_with_negatives(
            df_level, super_map, level=level
        )
        val_pos = sum(len(v["positive"]) for v in target_test_mapping.values())
        val_neg = sum(len(v["negative"]) for v in target_test_mapping.values())
        val_total = val_pos + val_neg

        # k-fold sizes if present
        kfold_sizes = _read_kfold_sizes(SAME_SUP_RESULTS, level) if args.include_kfold else None

        return {
            "level": level,
            "num_proteins": num_proteins,
            "num_subfamilies": num_subfamilies,
            "num_families": num_families,
            "basic_train": basic_train,
            "basic_test": basic_test,
            "val_pos": val_pos,
            "val_neg": val_neg,
            "val_total": val_total,
            "kfold": kfold_sizes,
        }

    levels: List[str] = ["subfamily", "family"] if args.level == "both" else [args.level]
    stats: List[Dict[str, any]] = [compute_for_level(lvl) for lvl in levels]

    # Print human-readable summary
    print("=== Dataset Statistics ===")
    print(f"Data path: {os.path.abspath(data_path)}")
    for s in stats:
        print(f"\n-- {s['level'].capitalize()} Level --")
        print(f"Number of Proteins: {s['num_proteins']}")
        print(f"Number of Subfamily: {s['num_subfamilies']}")
        print(f"Number of Family: {s['num_families']}")
        print(f"Train/Test (custom_split): train={s['basic_train']}, test={s['basic_test']}")
        print(
            f"Validation (same-superfamily): pos={s['val_pos']}, neg={s['val_neg']}, total={s['val_total']}"
        )
        if s["kfold"]:
            print(
                "K-Fold sizes (per fold): train="
                + ", ".join(map(str, s["kfold"]["num_train"]))
                + "; val="
                + ", ".join(map(str, s["kfold"]["num_val"]))
            )

    # Build Markdown table similar to the second image
    md_sections: List[str] = []
    for s in stats:
        headers = [
            f"{s['level'].capitalize()} Level",
            "#Families",
            "#Subfamilies",
            "#Proteins",
            "Train",
            "Test",
            "Validation (pos)",
            "Validation (neg)",
            "Validation (total)",
        ]
        rows = [[
            s["level"],
            str(s["num_families"]),
            str(s["num_subfamilies"]),
            str(s["num_proteins"]),
            str(s["basic_train"]),
            str(s["basic_test"]),
            str(s["val_pos"]),
            str(s["val_neg"]),
            str(s["val_total"]),
        ]]
        md_sections.append(_markdown_table(headers, rows))

        if s["kfold"]:
            k_headers = ["K-Fold", *[f"Fold {i+1} Train" for i in range(len(s["kfold"]["num_train"]))], *[f"Fold {i+1} Val" for i in range(len(s["kfold"]["num_val"]))]]
            k_rows = [[
                "Sizes",
                *[str(x) for x in s["kfold"]["num_train"]],
                *[str(x) for x in s["kfold"]["num_val"]],
            ]]
            md_sections.append(_markdown_table(k_headers, k_rows))

    markdown = "\n\n".join(md_sections)

    if args.save_markdown:
        os.makedirs(SAME_SUP_RESULTS, exist_ok=True)
        out_md = os.path.join(SAME_SUP_RESULTS, "dataset_summary.md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"\nSaved Markdown summary to: {out_md}")
    else:
        print("\nMarkdown Summary:\n")
        print(markdown)


if __name__ == "__main__":
    main()


