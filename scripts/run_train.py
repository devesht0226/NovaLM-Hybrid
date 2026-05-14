import sys

from src.training.train import main


if __name__ == "__main__":
    # Provide convenience defaults unless explicitly supplied.
    if "--tokens_json" not in sys.argv:
        sys.argv.extend(["--tokens_json", "data/processed/token_ids.json"])
    if "--out_dir" not in sys.argv:
        sys.argv.extend(["--out_dir", "artifacts/checkpoints"])
    main()
