"""Entrypoint scaffolding for Reddit streaming producers."""

from __future__ import annotations
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import httpx

import fasttext
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.streaming.producer.producer import Producer
from src.streaming.producer.reddit.base import (
    build_watchlist_submission,
    build_watchlist_subreddit,
    CookieManager,
    run_tracking_pipeline,
)


SRC_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = SRC_DIR / "model"
REDDIT_DIR = Path(__file__).resolve().parent
COOKIE_GLOB = REDDIT_DIR / "cookies" / "cookies-*.pkl"

LANGUAGE_MODEL_PATH = MODEL_DIR / "lid.176.bin"
RELEVANCE_MODEL_DIR = MODEL_DIR / "relevance-roberta-base"
RELEVANCE_THRESHOLD_PATH = RELEVANCE_MODEL_DIR / "threshold.json"
SENTIMENT_MODEL_1_DIR = MODEL_DIR / "sentiment-twitter-roberta-base"
SENTIMENT_MODEL_2_DIR = MODEL_DIR / "sentiment-cryptobert"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
PROXY = "http://zvjcd_hongh:1SthAX9d@117.0.113.155:19108"

REDDIT_STATUS_TOPIC = "reddit-status"
REDDIT_COMMENTS_TOPIC = "reddit-comments"
REDDIT_SUBMISSIONS_TOPIC = "reddit-submissions"
RUN_INTERVAL_MINUTES = 20




def main() -> None:
    """Initialize Reddit producer dependencies and bootstrap in-memory state from MinIO."""
    with RELEVANCE_THRESHOLD_PATH.open("r", encoding="utf-8") as threshold_file:
        threshold_payload = json.load(threshold_file)

    threshold = threshold_payload["optimal_threshold"]
    model_language = fasttext.load_model(str(LANGUAGE_MODEL_PATH))

    tokenizer_relevance = AutoTokenizer.from_pretrained(str(RELEVANCE_MODEL_DIR))
    model_relevance = AutoModelForSequenceClassification.from_pretrained(str(RELEVANCE_MODEL_DIR))
    model_relevance.to(DEVICE)
    model_relevance.eval()

    tokenizer_sentiment_1 = AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_1_DIR))
    model_sentiment_1 = AutoModelForSequenceClassification.from_pretrained(str(SENTIMENT_MODEL_1_DIR))
    model_sentiment_1.to(DEVICE)
    model_sentiment_1.eval()

    tokenizer_sentiment_2 = AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_2_DIR))
    model_sentiment_2 = AutoModelForSequenceClassification.from_pretrained(str(SENTIMENT_MODEL_2_DIR))
    model_sentiment_2.to(DEVICE)
    model_sentiment_2.eval()

    submission_producer = Producer(topic=REDDIT_SUBMISSIONS_TOPIC)
    comment_producer = Producer(topic=REDDIT_COMMENTS_TOPIC)
    status_producer = Producer(topic=REDDIT_STATUS_TOPIC)
    watchlist_subreddit = build_watchlist_subreddit()
    watchlist_submission = build_watchlist_submission(watchlist_subreddit)

    cookie_manager = CookieManager(str(COOKIE_GLOB))
    
    client = httpx.AsyncClient(proxy=PROXY, timeout=30.0)
    
    cookie_manager.reset()
    run_tracking_pipeline(
        client=client,
        watchlist_subreddit=watchlist_subreddit,
        watchlist_submission=watchlist_submission,
        cookie_manager=cookie_manager,
        first_run=True,
        model_language=model_language,
        tokenizer_relevance=tokenizer_relevance,
        model_relevance=model_relevance,
        threshold=threshold,
        tokenizer_sentiment_1=tokenizer_sentiment_1,
        tokenizer_sentiment_2=tokenizer_sentiment_2,
        model_sentiment_1=model_sentiment_1,
        model_sentiment_2=model_sentiment_2,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        submission_producer=submission_producer,
        comment_producer=comment_producer,
    )
    status_producer.send(
        {
            "first_run": True,
            "timestamp_utc": int(datetime.now(timezone.utc).timestamp())
        },
        topic=REDDIT_STATUS_TOPIC,
        key="reddit-producer",
    )
    status_producer.flush()
    while True:
        now = datetime.now(timezone.utc)
        next_run = (
            now.replace(second=0, microsecond=0)
            + timedelta(minutes=RUN_INTERVAL_MINUTES - now.minute % RUN_INTERVAL_MINUTES)
        )
        while True:
            now = datetime.now(timezone.utc)
            if now >= next_run:
                break
            remaining = (next_run - now).total_seconds()

            if remaining > 60:
                time.sleep(30)
            elif remaining > 30:
                time.sleep(10)
            elif remaining > 10:
                time.sleep(5)
            elif remaining > 1:
                time.sleep(0.5)
            else:
                time.sleep(0.01)
        cookie_manager.reset()
        run_tracking_pipeline(
            client=client,
            watchlist_subreddit=watchlist_subreddit,
            watchlist_submission=watchlist_submission,
            cookie_manager=cookie_manager,
            first_run=False,
            model_language=model_language,
            tokenizer_relevance=tokenizer_relevance,
            model_relevance=model_relevance,
            threshold=threshold,
            tokenizer_sentiment_1=tokenizer_sentiment_1,
            tokenizer_sentiment_2=tokenizer_sentiment_2,
            model_sentiment_1=model_sentiment_1,
            model_sentiment_2=model_sentiment_2,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            submission_producer=submission_producer,
            comment_producer=comment_producer,
        )
        status_producer.send(
                {
                    "first_run": False,
                    "timestamp_utc": int(next_run.timestamp()),
                },
                topic=REDDIT_STATUS_TOPIC,
                key="reddit-producer",
            )
        status_producer.flush()

if __name__ == "__main__":
    main()
