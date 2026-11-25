
from datasets import load_dataset

print("Downloading Yelp review dataset...")

dataset = load_dataset("yelp_review_full")

print("\nDataset downloaded successfully!")

print(dataset)
