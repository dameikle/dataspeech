import argparse
from datasets import load_dataset


def filter_valid_audio(example):
    text_valid = example['text'] not in ['--', '', None]
    duration_valid = example['duration'] > 0.1
    filesize_valid = example['wav_filesize'] > 2000
    return text_valid and duration_valid and filesize_valid


def load_and_filter_dataset(args):

    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration, split=args.split)
    else:
        dataset = load_dataset(args.dataset_name, split=args.split)

    print(f"Loaded {args.split} split. Original size: {len(dataset)}")

    if args.num_records is not None:
        dataset = dataset.select(range(min(args.num_records, len(dataset))))
        print(f"Limited to {len(dataset)} records")

    filtered_dataset = dataset.filter(
        filter_valid_audio,
        num_proc=args.num_proc
    )

    print(f"Filtered {args.split} split size: {len(filtered_dataset)}")
    print(f"Removed {len(dataset) - len(filtered_dataset)} examples")

    return filtered_dataset


def main():
    parser = argparse.ArgumentParser(description="Filter bad audio from dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--configuration", type=str, default=None, help="Configuration of the dataset (optional)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (default: train)")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for filtering (default: 8)")
    parser.add_argument("--output_dir", type=str, default="./filtered_dataset",
                        help="Directory to save the filtered dataset")
    parser.add_argument("--num_records", type=int, default=None,
                        help="Number of records to process (optional, processes all if not specified)")
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry run without saving the dataset")

    args = parser.parse_args()

    filtered_dataset = load_and_filter_dataset(args)

    if args.dry_run:
        print("Dry run completed. Dataset not saved.")
        print("Sample of filtered dataset:")
        for i, example in enumerate(filtered_dataset.select(range(min(5, len(filtered_dataset))))):
            print(f"Example {i + 1}:")
            print(example)
            print()
    else:
        if args.output_dir:
            print("Saving to disk...")
            filtered_dataset.save_to_disk(f"{args.output_dir}/{args.split}")
            print(f"Saved filtered dataset to {args.output_dir}/{args.split}")
        if args.repo_id:
            print("Pushing to the hub...")
            if args.configuration:
                filtered_dataset.push_to_hub(args.repo_id, args.configuration, split=args.split)
            else:
                filtered_dataset.push_to_hub(args.repo_id, split=args.split)
            print(f"Filtered dataset pushed to hub at {args.output_dir}/{args.split}")


if __name__ == "__main__":
    main()