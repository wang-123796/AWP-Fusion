import os
import sys
import pandas as pd
import torch
import esm


def create_dir_if_not_exist(dir_path):
    """Create directory if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def extract_esm_features(csv_path, save_dir, model, alphabet, device, repr_layer=33):
    """
    Extract ESM features from sequences in a CSV file and save them as .pt files.

    The CSV file must contain two columns: 'uniprot_id' and 'sequence'.

    :param csv_path: path to the input CSV file
    :param save_dir: directory to save the extracted features
    :param model: ESM model
    :param alphabet: ESM alphabet
    :param device: computation device (cpu or cuda)
    :param repr_layer: layer index for representation extraction (33 for esm2_t33_650M_UR50D)
    """
    df = pd.read_csv(csv_path)
    create_dir_if_not_exist(save_dir)

    batch_converter = alphabet.get_batch_converter()

    with torch.no_grad():
        for idx, row in df.iterrows():
            uniprot_id = row["uniprot_id"]
            sequence = row["sequence"]

            save_path = os.path.join(save_dir, f"{uniprot_id}.pt")
            if os.path.exists(save_path):
                print(f"Skipping {uniprot_id}: feature already exists")
                continue

            if pd.isna(sequence) or len(sequence) < 1:
                print(f"Warning: invalid sequence for {uniprot_id}, skipping")
                continue

            batch_data = [(str(uniprot_id), sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            outputs = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_embeddings = outputs["representations"][repr_layer]
            # Remove <cls> and <eos> tokens
            sequence_embeddings = token_embeddings[0, 1:-1, :]

            torch.save(sequence_embeddings.cpu(), save_path)
            print(f"Saved features for {uniprot_id} to {save_path}")


def parse_task_and_split(csv_filename):
    """
    Parse task name and split from a CSV filename.

    Supported formats:
        train_pH.csv    -> ("pH", "train")
        test_Tm.csv     -> ("Tm", "test")
        train_Topt.csv  -> ("Topt", "train")
        test_Topt.csv   -> ("Topt", "test")

    :param csv_filename: CSV filename (with or without path)
    :return: (task, split) or (None, None) if parsing fails
    """
    base = os.path.basename(csv_filename)
    name, _ = os.path.splitext(base)

    if "_" not in name:
        return None, None

    split, task = name.split("_", 1)

    if split not in ("train", "test"):
        return None, None

    return task, split


if __name__ == "__main__":
    # Global configuration
    REPR_LAYER = 33
    PROCESSED_DIR = os.path.join("data", "processed")
    MODEL_PATH = os.path.join("data", "esm2_t33_650M_UR50D.pt")

    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <csv_file> [<csv_file> ...]")
        print("Example:")
        print("  python extract_features.py test_pH.csv")
        print("  python extract_features.py train_Topt.csv test_Topt.csv")
        sys.exit(1)

    csv_files = sys.argv[1:]

    # Parse filenames and determine output directories
    jobs = []
    for csv_file in csv_files:
        task, split = parse_task_and_split(csv_file)

        if task is None:
            print(f"Error: cannot parse task and split from '{csv_file}'.")
            print("Use naming convention train_<task>.csv or test_<task>.csv.")
            sys.exit(1)

        if not os.path.exists(csv_file):
            print(f"Error: file not found: {csv_file}")
            sys.exit(1)

        save_dir = os.path.join(PROCESSED_DIR, f"{task}_sequence_{split}_feat")
        jobs.append((csv_file, save_dir, task, split))

    # Load ESM model from local weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model weights not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Process each CSV file
    for csv_file, save_dir, task, split in jobs:
        print(f"\n{'=' * 50}")
        print(f"Processing: {csv_file}")
        print(f"Task: {task} | Split: {split}")
        print(f"Output directory: {save_dir}")
        print(f"{'=' * 50}")

        extract_esm_features(
            csv_file, save_dir,
            model, alphabet, device, REPR_LAYER
        )

    print("\nAll done!")