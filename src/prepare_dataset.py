
from datasets import load_dataset
import json
import random
from pathlib import Path

# Number of reviews required for each star rating
sample_targets = {
    0: 750,
    1: 750,
    2: 1500,
    3: 750,
    4: 750
}

# Map star ratings to sentiment categories
sentiment_mapping = {
    0: "negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "positive"
}

# Initialize sample storage
samples = {
    rating: [] for rating in range(5)
}

print("Connecting to Yelp dataset...")

dataset = load_dataset(
    "Yelp/yelp_review_full",
    split="train",
    streaming=True
)

dataset = dataset.shuffle(
    seed=42,
    buffer_size=20000
)

print("Collecting balanced reviews...")

for item in dataset:
    rating = item["label"]

    if len(samples[rating]) < sample_targets[rating]:
        samples[rating].append(item["text"])

    if all(
        len(samples[rating]) == sample_targets[rating]
        for rating in range(5)
    ):
        break

# Convert reviews into JSON records
reviews = []
review_id = 1

for rating, texts in samples.items():
    for review in texts:
        reviews.append({
            "id": review_id,
            "review": review,
            "stars": rating + 1,
            "sentiment": sentiment_mapping[rating]
        })

        review_id += 1

# Shuffle final dataset
random.Random(42).shuffle(reviews)

# Save JSON dataset
output_path = Path("dataset/reviews.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        reviews,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nDataset preparation completed!")
print(f"Total reviews: {len(reviews)}")
print(f"Saved to: {output_path.resolve()}")

for sentiment in ["negative", "neutral", "positive"]:
    count = sum(
        1 for review in reviews
        if review["sentiment"] == sentiment
    )

    print(f"{sentiment.capitalize()}: {count}")
