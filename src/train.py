import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "reviews.json"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place your reviews.json file in the project's dataset folder."
        )

    df = pd.read_json(DATA_PATH)
    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")

    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["review"] = df["review"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    df = df[df["sentiment"].isin(["negative", "neutral", "positive"])]
    df = df[df["review"].str.strip().ne("")]
    if df["sentiment"].nunique() < 3:
        raise ValueError("Expected all three sentiment classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"],
        test_size=0.20, random_state=42, stratify=df["sentiment"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print(f"Training on {len(X_train):,} reviews...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    report = classification_report(
        y_test, predictions,
        labels=["negative", "neutral", "positive"],
        output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "test_samples": int(len(y_test)),
        "classes": ["negative", "neutral", "positive"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_test, predictions,
            labels=["negative", "neutral", "positive"]
        ).tolist()
    }

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "sentiment_pipeline.joblib")
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nEvaluation results")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Model saved to: {MODEL_DIR / 'sentiment_pipeline.joblib'}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'model_metrics.json'}")

if __name__ == "__main__":
    main()
