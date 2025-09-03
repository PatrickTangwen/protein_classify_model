import argparse
import random
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

import config_same_sup as config
from data_loader import load_protein_data, load_superfamily_map
from feature_engineering import build_features
from data_splitting_same_sup import generate_negative_controls


def build_folds(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return list(kf.split(np.zeros_like(y), y))


def main():
    parser = argparse.ArgumentParser(description="Print per-fold protein counts for 3-fold CV")
    parser.add_argument('--level', type=str, required=True, choices=['subfamily', 'family'], help='Classification level')
    args = parser.parse_args()

    df = load_protein_data(config.PROTEIN_DATA_PATH, level=args.level)
    superfamily_map = load_superfamily_map(config.SUPERFAMILY_MAP_PATH)

    # Build labels (consistent with fold_script)
    _, y, label_encoder, _, _ = build_features(
        df,
        level=args.level,
        max_domains=config.MAX_DOMAINS,
        max_separators=config.MAX_SEPARATORS,
        evalue_threshold=config.EVALUE_THRESHOLD,
    )

    folds = build_folds(y)
    print(f"Prepared {len(folds)} stratified folds (random_state=42)")
    print(f"Total proteins: {len(df)} | Num classes: {len(label_encoder.classes_)}\n")

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print("=" * 70)
        print(f"Fold {fold_idx}")
        print(f"Train proteins: {len(train_idx)}")
        print(f"Validation proteins (original fold): {len(test_idx)}")

        # Reproduce negative control generation exactly as fold_script
        random.seed(42 + fold_idx)
        negative_control_dict, target_to_test_indices = generate_negative_controls(
            df, test_idx.tolist(), train_idx.tolist(), superfamily_map, args.level
        )
        final_test_indices = [idx for indices in target_to_test_indices.values() for idx in indices]
        all_negative_indices = set()
        for _, negs in negative_control_dict.items():
            all_negative_indices.update(negs)

        test_indices_with_negatives = set(final_test_indices).union(all_negative_indices)

        print(f"  Positives retained (with superfamily mapping): {len(final_test_indices)}")
        print(f"  Negative controls added: {len(all_negative_indices)}")
        print(f"  Validation total (pos + negatives): {len(test_indices_with_negatives)}")

    print("\nDone.")


if __name__ == '__main__':
    main()


