"""Base utilities and pipelines for Reddit streaming producers."""
import asyncio
import glob
import html
import logging
import random
import re
import pickle
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import emoji
import httpx
import numpy as np
import polars as pl
import torch

from src.utils.s3_client import MinIOWriter


logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s UTC - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

WATCHLIST_SUBREDDITS = [
    "Bitcoin",
    "InBitcoinWeTrust",
    "btc",
    "Buttcoin",
    "BitcoinBeginners",
    "BitcoinMarkets",
    "BitcoinMining",
    "BitcoinDE",
    "BitcoinBrasil",
    "BitcoinCA",
    "BitcoinUK",
    "BitcoinEU",
    "Bitcoincash",
    "BitcoinIndia",
    "BitcoinAUS",
    "bitcoincashSV",
    "Daytrading",
    "CryptoCurrency",
    "CryptoMarkets",
    "Trading",
    "BitMartExchange",
    "XGramatikInsights",
    "CryptoChartWatch",
    "CryptoIndia",
    "CryptoTax",
    "DubaiCrypto",
    "CryptoCurrencyClassic",
    "cryptocurrencymemes",
    "CryptoHelp",
    "CryptoExchange",
    "binance",
    "CryptoReality",
    "AllCryptoBets",
    "CryptoNews",
    "CryptoTechnology",
    "nanotrade",
    "WallStreetBetsCrypto",
    "Crypto_com",
    "BinanceCrypto",
    "Crypto_Currency_News",
    "CryptoStock",
    "altcoin",
    "CryptoMarsShots",
    "Crypto_General",
    "CryptoTradingFloor",
    "CryptoMars",
    "CryptoInvesting",
    "CryptoMoon",
    "CryptoMoonInvestors",
    "CryptoCurrencyTrading",
    "HodlyCrypto",
    "CryptoNews2day",
]
def build_watchlist_subreddit():
    """Create the default subreddit watchlist state."""
    return {
        subreddit: {
            "checked": False,
            "latest_submission_id": "",
        }
        for subreddit in WATCHLIST_SUBREDDITS
    }


def get_latest_minio_paths(minio_client: MinIOWriter, folder_path: str, prefix: str) -> list[str]:
    """Return the two newest parquet object paths in MinIO matching the monthly Reddit naming pattern."""
    pattern = re.compile(rf"^{re.escape(prefix)}_\d{{4}}-\d{{2}}\.parquet$")
    object_paths = minio_client.list_objects(prefix=folder_path, recursive=True)
    matched_paths = [path for path in object_paths if pattern.match(Path(path).name)]
    return sorted(matched_paths)[-2:]


def build_watchlist_submission(watchlist_subreddit, minio_client=None):
    """Build submission watchlist state from the latest Reddit parquet files in MinIO."""
    logger.info("[reddit] watchlist_submission start")
    minio_client = minio_client or MinIOWriter(bucket="reddit")

    submission_paths = get_latest_minio_paths(minio_client, "submissions/", "RS")
    comment_paths = get_latest_minio_paths(minio_client, "comments/", "RC")

    submission_frames = []
    for path in submission_paths:
        df = minio_client.read_parquet(path)
        if df is not None and not df.is_empty():
            submission_frames.append(df.select(["id", "created_utc", "subreddit", "relevance"]))

    if not submission_frames:
        return {}

    comment_frames = []
    for path in comment_paths:
        df = minio_client.read_parquet(path)
        if df is not None and not df.is_empty():
            comment_frames.append(
                df.select(
                    [
                        pl.col("id").alias("comment_id"),
                        "link_id",
                        "created_utc",
                    ]
                )
            )

    df_sub = pl.concat(submission_frames, how="vertical_relaxed")
    df_comm = pl.concat(comment_frames, how="vertical_relaxed") if comment_frames else pl.DataFrame()

    max_sub = df_sub.select(pl.col("created_utc").max()).item()
    max_comm = df_comm.select(pl.col("created_utc").max()).item() if not df_comm.is_empty() else None
    max_utc = max_sub if max_comm is None else max(max_sub, max_comm)

    df_offsets = (
        df_sub.lazy()
        .filter(pl.col("subreddit").is_in(list(watchlist_subreddit.keys())))
        .sort("created_utc", descending=True)
        .group_by("subreddit")
        .first()
        .select(["subreddit", "id"])
        .collect()
    )
    for subreddit, latest_id in zip(df_offsets["subreddit"], df_offsets["id"]):
        watchlist_subreddit[subreddit]["latest_submission_id"] = latest_id

    df_sub_filtered = df_sub.filter(pl.col("created_utc") >= (max_utc - 30 * 24 * 60 * 60))

    if df_comm.is_empty():
        df_final = (
            df_sub_filtered.with_columns(
                [
                    pl.concat_list([pl.col("created_utc")]).alias("list_active_time"),
                    pl.lit([], dtype=pl.List(pl.String)).alias("list_comment_id"),
                ]
            )
            .select(["id", "list_active_time", "list_comment_id", "relevance"])
        )
    else:
        submission_ids = df_sub_filtered.get_column("id").to_list()
        df_comm_grouped = (
            df_comm.lazy()
            .filter(pl.col("link_id").is_in([f"t3_{sub_id}" for sub_id in submission_ids]))
            .with_columns(pl.col("link_id").str.replace(r"^t3_", "").alias("submission_id"))
            .sort("created_utc")
            .group_by("submission_id")
            .agg(
                [
                    pl.col("created_utc").alias("comm_utcs"),
                    pl.col("comment_id").alias("comm_ids"),
                ]
            )
            .collect()
        )

        df_final = (
            df_sub_filtered.join(df_comm_grouped, left_on="id", right_on="submission_id", how="left")
            .with_columns(
                [
                    pl.concat_list(
                        [
                            pl.concat_list([pl.col("created_utc")]),
                            pl.col("comm_utcs").fill_null(pl.lit([], dtype=pl.List(pl.Int64))),
                        ]
                    ).alias("list_active_time"),
                    pl.col("comm_ids").fill_null(pl.lit([], dtype=pl.List(pl.String))).alias("list_comment_id"),
                ]
            )
            .select(["id", "list_active_time", "list_comment_id", "relevance"])
        )

    watchlist_submission = {
        row["id"]: {
            "list_active_time": row["list_active_time"],
            "list_comment_id": row["list_comment_id"],
            "relevance": row["relevance"],
            "active": is_active(row["list_active_time"], EPSILON, BETA, max_utc),
            "checked": False,
        }
        for row in df_final.to_dicts()
    }

    for sub_id in list(watchlist_submission.keys()):
        list_active_time = watchlist_submission[sub_id]["list_active_time"]
        oldest_active_time = min(list_active_time) if list_active_time else max_utc
        if (
            not watchlist_submission[sub_id]["active"]
            and (max_utc - oldest_active_time) > (30 * 24 * 60 * 60)
        ):
            del watchlist_submission[sub_id]

    active_submission_count = sum(
        1 for submission in watchlist_submission.values() if submission.get("active")
    )
    logger.info(f"[reddit] watchlist_submission done: {active_submission_count}")
    return watchlist_submission


def publish_records(records, producer, topic, key_field="id"):
    """Publish a batch of enriched records to Redpanda."""
    if not records or producer is None:
        return

    for record in records:
        record_key = record.get(key_field)
        key = str(record_key) if record_key is not None else None
        producer.send(record, topic=topic, key=key)

    producer.flush()


def _coerce_string(value):
    """Convert a value to nullable string semantics."""
    if value is None:
        return None
    return str(value)


def _coerce_int(value):
    """Convert a value to nullable integer semantics."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return int(float(text))


SUBMISSION_STRING_FIELDS = {
    "id",
    "author",
    "subreddit",
    "link_flair_text",
    "title",
    "selftext",
    "language",
}
SUBMISSION_INT_FIELDS = {
    "created_utc",
    "relevance",
    "bot",
    "sentiment",
}
def transform_submission_record(record):
    """Normalize a submission record before publishing to Redpanda."""
    transformed = dict(record)
    for field in SUBMISSION_STRING_FIELDS:
        transformed[field] = _coerce_string(transformed.get(field))
    for field in SUBMISSION_INT_FIELDS:
        transformed[field] = _coerce_int(transformed.get(field))
    return transformed


COMMENT_STRING_FIELDS = {
    "id",
    "author",
    "link_id",
    "body",
    "language",
}
COMMENT_INT_FIELDS = {
    "created_utc",
    "relevance",
    "bot",
    "sentiment",
}
def transform_comment_record(record):
    """Normalize a comment record before publishing to Redpanda."""
    transformed = dict(record)
    for field in COMMENT_STRING_FIELDS:
        transformed[field] = _coerce_string(transformed.get(field))
    for field in COMMENT_INT_FIELDS:
        transformed[field] = _coerce_int(transformed.get(field))
    return transformed


def get_user_agent():
    chrome_versions = [
        "124.0.0.0", "123.0.0.0", "122.0.0.0", "121.0.0.0", "120.0.0.0",
        "119.0.0.0", "118.0.0.0", "117.0.0.0", "116.0.0.0", "115.0.0.0",
        "114.0.0.0", "113.0.0.0", "112.0.0.0", "111.0.0.0", "110.0.0.0",
        "109.0.0.0", "108.0.0.0", "107.0.0.0", "106.0.0.0", "105.0.0.0",
    ]
    firefox_versions = [
        "125.0", "124.0", "123.0", "122.0", "121.0",
        "120.0", "119.0", "118.0", "117.0", "116.0",
        "115.0", "114.0", "113.0", "112.0", "111.0",
        "110.0", "109.0", "108.0", "107.0", "106.0",
    ]
    safari_versions = [
        "17.4.1", "17.3.1", "17.2.1", "17.1.2", "17.0",
        "16.6", "16.5", "16.4.1", "16.3", "16.2",
        "15.6.1", "15.5", "15.4", "15.3", "15.2",
        "14.1.2", "14.1.1", "14.0.3", "14.0.2", "14.0.1",
    ]
    
    windows_versions = [
        "Windows NT 10.0; Win64; x64",
        "Windows NT 10.0; WOW64",
        "Windows NT 6.1; Win64; x64",
        "Windows NT 6.3; Win64; x64",
    ]
    mac_versions = [
        "Macintosh; Intel Mac OS X 10_15_7",
        "Macintosh; Intel Mac OS X 10_15_6",
        "Macintosh; Intel Mac OS X 10_14_6",
        "Macintosh; Intel Mac OS X 11_6_8",
        "Macintosh; Intel Mac OS X 12_6_3",
        "Macintosh; Intel Mac OS X 13_4_1",
        "Macintosh; Intel Mac OS X 14_4_1",
        "Macintosh; Intel Mac OS X 14_3_1",
    ]
    linux_versions = [
        "X11; Linux x86_64",
        "X11; Ubuntu; Linux x86_64",
        "X11; Fedora; Linux x86_64",
        "X11; Linux i686",
        "X11; Linux aarch64",
        "X11; Linux armv7l",
        "X11; Linux armv8l",
        "X11; CrOS x86_64 14541.0.0",
    ]
    
    browser = random.choice(["chrome", "firefox", "safari"])
    
    if browser == "chrome":
        os = random.choice(windows_versions + mac_versions + linux_versions)
        v = random.choice(chrome_versions)
        return f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{v} Safari/537.36"
    
    elif browser == "firefox":
        os = random.choice(windows_versions + mac_versions + linux_versions)
        v = random.choice(firefox_versions)
        return f"Mozilla/5.0 ({os}; rv:{v}) Gecko/20100101 Firefox/{v}"
    
    else:  
        os = random.choice(mac_versions)
        v = random.choice(safari_versions)
        return f"Mozilla/5.0 ({os}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{v} Safari/605.1.15"

class CookieManager:
    def __init__(self, cookie_pattern="cookies-*.pkl"):
        files = sorted(glob.glob(cookie_pattern))
        self.cookies = {}

        for file in files:
            raw_cookie = pickle.load(open(file, "rb"))
            cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in raw_cookie])
            self.cookies[file] = cookie_str

        self.watchlist = {f: 100 for f in files}
        logger.info(f"Loaded {len(self.watchlist)} cookies")

    def reset(self):
        self.watchlist = {cookie_file: 100 for cookie_file in self.cookies}
        logger.info("[reddit] cookie reset")

    def get_cookie(self):
        available = {f: v for f, v in self.watchlist.items() if v > 0}
        if not available:
            return None
        max_val = max(available.values())
        candidates = [f for f, v in available.items() if v == max_val]
        chosen = random.choice(candidates)
        self.watchlist[chosen] -= 1
        return self.cookies[chosen]

    def status(self):
        return sum(self.watchlist.values())


HTTP_RETRY_ATTEMPTS = 20
HTTP_RETRY_BASE_DELAY_SECONDS = 0.5
async def fetch_submissions(
    client=None,
    subreddit=None,
    limit=None,
    after=None,
    before=None,
    count=None,
    show=None,
    sr_detail=None,
    user_agent=None,
    cookie=None,
):
    """Fetch newest submissions for a subreddit via Reddit's JSON endpoint."""
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    headers = {
        "User-Agent": user_agent,
        "Cookie": cookie,
    }
    params = {}

    if limit is not None:
        params["limit"] = limit
    if after is not None:
        params["after"] = after
    if before is not None:
        params["before"] = before
    if count is not None:
        params["count"] = count
    if show is not None:
        params["show"] = show
    if sr_detail is not None:
        params["sr_detail"] = sr_detail

    response = None
    for attempt in range(1, HTTP_RETRY_ATTEMPTS + 1):
        try:
            response = await client.get(url, headers=headers, params=params)
            if response.status_code == 200:
                break
            if attempt >= 3:
                    logger.warning(
                        f"Request failed for r/{subreddit}: {response.status_code} "
                        f"(attempt {attempt}/{HTTP_RETRY_ATTEMPTS})"
                    )
        except Exception as exc:
            if attempt >= 3:
                logger.warning(
                    f"Network error for r/{subreddit}: {exc} "
                    f"(attempt {attempt}/{HTTP_RETRY_ATTEMPTS})"
                )

        if attempt >= HTTP_RETRY_ATTEMPTS:
            return None
        delay_seconds = HTTP_RETRY_BASE_DELAY_SECONDS * attempt
        await asyncio.sleep(delay_seconds)

    data = response.json()
    fields = [
        "id",
        "author",
        "created_utc",
        "subreddit",
        "link_flair_text",
        "title",
        "selftext",
    ]
    submissions = []
    if "data" in data and "children" in data["data"]:
        for item in data["data"]["children"]:
            post = item["data"]
            submissions.append({field: post.get(field) for field in fields})

    return submissions
async def fetch_comments(
    client=None,
    submission_id=None,
    depth=None,
    sort=None,
    user_agent=None,
    cookie=None,
    limit=None,
    showedits=None,
    showmedia=None,
    showmore=None,
    showtitle=None,
    threaded=None,
    sr_detail=None,
    children_ids=None,
    limit_children=None,
):
    """Fetch submission comments or `morechildren` expansions via Reddit JSON APIs."""
    headers = {
        "User-Agent": user_agent,
        "Cookie": cookie,
    }
    fields = ["id", "author", "created_utc", "link_id", "body"]

    if children_ids is None:
        url = f"https://www.reddit.com/comments/{submission_id}.json"
        params = {}
        if limit is not None:
            params["limit"] = limit
        if depth is not None:
            params["depth"] = depth
        if sort is not None:
            params["sort"] = sort
        if showedits is not None:
            params["showedits"] = showedits
        if showmedia is not None:
            params["showmedia"] = showmedia
        if showmore is not None:
            params["showmore"] = showmore
        if showtitle is not None:
            params["showtitle"] = showtitle
        if threaded is not None:
            params["threaded"] = threaded
        if sr_detail is not None:
            params["sr_detail"] = sr_detail

        response = None
        for attempt in range(1, HTTP_RETRY_ATTEMPTS + 1):
            try:
                response = await client.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    break
                if attempt >= 3:
                    logger.warning(
                        f"Request failed for comments of {submission_id}: "
                        f"{response.status_code} (attempt {attempt}/{HTTP_RETRY_ATTEMPTS})"
                    )
            except Exception as exc:
                if attempt >= 3:
                    logger.warning(
                        f"Network error for comments of {submission_id}: "
                        f"{exc} (attempt {attempt}/{HTTP_RETRY_ATTEMPTS})"
                    )

            if attempt >= HTTP_RETRY_ATTEMPTS:
                return None, None
            delay_seconds = HTTP_RETRY_BASE_DELAY_SECONDS * attempt
            await asyncio.sleep(delay_seconds)
        data = response.json()
        comments = []
        more_children_ids = []
        try:
            children = data[1]["data"]["children"]
            for item in children:
                kind = item.get("kind")
                if kind == "t1":
                    comment = item["data"]
                    comments.append({field: comment.get(field) for field in fields})
                elif kind == "more":
                    more_children_ids.extend(item["data"].get("children", []))
        except Exception:
            pass
        return comments, more_children_ids
    else: 
        url = "https://www.reddit.com/api/morechildren.json"
        params = {
            "children": ",".join(children_ids),
            "api_type": "json",
        }
        if submission_id is not None:
            params["link_id"] = f"t3_{submission_id}"
        if sort is not None:
            params["sort"] = sort
        if limit_children is not None:
            params["limit_children"] = limit_children
        if depth is not None:
            params["depth"] = depth

        response = None
        for attempt in range(1, HTTP_RETRY_ATTEMPTS + 1):
            try:
                response = await client.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    break
                if attempt >= 3:
                    logger.warning(
                        f"Request failed for morechildren of {submission_id}: "
                        f"{response.status_code} (attempt {attempt}/{HTTP_RETRY_ATTEMPTS})"
                    )
            except Exception as exc:
                if attempt >= 3:
                    logger.warning(
                        f"Network error for morechildren of {submission_id}: "
                        f"{exc} (attempt {attempt}/{HTTP_RETRY_ATTEMPTS})"
                    )

            if attempt >= HTTP_RETRY_ATTEMPTS:
                return None
            delay_seconds = HTTP_RETRY_BASE_DELAY_SECONDS * attempt
            await asyncio.sleep(delay_seconds)

        data = response.json()
        comments = []
        try:
            things = data["json"]["data"]["things"]
            for item in things:
                if item.get("kind") == "t1":
                    comment = item["data"]
                    comments.append({field: comment.get(field) for field in fields})
        except Exception:
            pass
        return comments
    

TITLE_BLOCKED_TEXTS = {
    "[deleted by user]",
    "[Removed by moderator]",
    "[Removed by Reddit]",
}
SELFTEXT_BLOCKED_TEXTS = {
    "[removed]",
    "[deleted]",
    "[Removed by Reddit on account of violating the [content policy](/help/contentpolicy).]",
}
BODY_BLOCKED_TEXTS = {
    "[removed]",
    "[deleted]",
}
def clean_text(text, blocked_texts=None, remove_tokens=False, lower=False, use_demoji=False):
    """Normalize and sanitize Reddit text for downstream inference."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = (
        emoji.replace_emoji(text, replace=" ")
        if remove_tokens
        else (emoji.demojize(text, delimiters=(" [", "] ")) if use_demoji else text)
    )
    text = re.sub(r"(?:https?://|www\.)[^\s)\]\}]+", " " if remove_tokens else " [URL] ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " " if remove_tokens else " [MEDIA] ", text)
    text = re.sub(
        r"(?<!\w)(?:r|u)/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)+",
        " " if remove_tokens else " [REDDIT_PATH] ",
        text,
    )
    text = re.sub(r"(?<!\w)u/[A-Za-z0-9_-]+", " " if remove_tokens else " [USER] ", text)
    text = re.sub(r"(?<!\w)r/[A-Za-z0-9_-]+", " " if remove_tokens else " [SUBREDDIT] ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"(?<=[A-Za-z0-9])[—–-](?=[A-Za-z0-9])", "-", text)
    text = re.sub(r"[\^\*\~\\/_—–#>|]", " ", text)
    text = re.sub(r"(?<![A-Za-z0-9])-(?![A-Za-z0-9]|\s*\d)|(?<=[A-Za-z0-9])-(?![A-Za-z0-9])", " ", text)
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", " ", text)
    text = re.sub(r"([\[\(\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\]\)\}])", r"\1", text)
    text = re.sub(r"\s+", " ", text).replace("\u200b", "").strip()
    if blocked_texts and text in blocked_texts:
        return ""
    if lower:
        text = text.lower()
    return text

    
def detect_submissions_language(submissions, model):
    """Detect submission language with a FastText-like model."""
    if not submissions:
        return []

    texts_to_predict = []
    indices_to_predict = []
    results = [None] * len(submissions)

    for idx, submission in enumerate(submissions):
        title = submission.get("title")
        selftext = submission.get("selftext")

        language_title = (
            clean_text(title, blocked_texts=TITLE_BLOCKED_TEXTS, remove_tokens=True, lower=True)
            if title
            else ""
        )
        language_selftext = (
            clean_text(selftext, blocked_texts=SELFTEXT_BLOCKED_TEXTS, remove_tokens=True, lower=True)
            if selftext
            else ""
        )

        lang_parts = []
        if language_title:
            lang_parts.append(language_title)
        if language_selftext:
            lang_parts.append(language_selftext)

        text_language = " ".join(lang_parts).strip()
        if not text_language or not any(ch.isalpha() for ch in text_language):
            results[idx] = "empty"
        else:
            texts_to_predict.append(text_language.replace("\n", " "))
            indices_to_predict.append(idx)

    if texts_to_predict:
        try:
            labels, _ = model.predict(texts_to_predict, k=1)
            for predict_idx, raw_label in enumerate(labels):
                predicted_lang = raw_label[0].replace("__label__", "")
                original_idx = indices_to_predict[predict_idx]
                results[original_idx] = "en" if predicted_lang == "en" else "non-en"
        except Exception as exc:
            logger.error(f"Language prediction error (submissions): {exc}")

    return results


def detect_comments_language(comments, model):
    """Detect comment language with a FastText-like model."""
    if not comments:
        return []

    texts_to_predict = []
    indices_to_predict = []
    results = [None] * len(comments)

    for idx, comment in enumerate(comments):
        body = comment.get("body")
        text_language = (
            clean_text(body, blocked_texts=BODY_BLOCKED_TEXTS, remove_tokens=True, lower=True)
            if body
            else ""
        )
        if not text_language or not any(ch.isalpha() for ch in text_language):
            results[idx] = "empty"
        else:
            texts_to_predict.append(text_language.replace("\n", " "))
            indices_to_predict.append(idx)

    if texts_to_predict:
        try:
            labels, _ = model.predict(texts_to_predict, k=1)
            for predict_idx, raw_label in enumerate(labels):
                predicted_lang = raw_label[0].replace("__label__", "")
                original_idx = indices_to_predict[predict_idx]
                results[original_idx] = "en" if predicted_lang == "en" else "non-en"
        except Exception as exc:
            logger.error(f"Language prediction error (comments): {exc}")

    return results


def detect_submissions_relevance(submissions, tokenizer, model, threshold, device, max_length=256, batch_size=16):
    """Predict submission relevance with a transformer classifier."""
    if not submissions:
        return []

    valid_texts = []
    valid_indices = []
    results = [None] * len(submissions)

    for idx, submission in enumerate(submissions):
        lang = submission.get("language")
        title_raw = submission.get("title") or ""
        flair_raw = submission.get("link_flair_text") or ""
        selftext_raw = submission.get("selftext") or ""

        is_english = lang == "en"
        is_emoji_only = lang == "empty" and (
            emoji.emoji_count(title_raw) > 0
            or emoji.emoji_count(flair_raw) > 0
            or emoji.emoji_count(selftext_raw) > 0
        )

        if is_english or is_emoji_only:
            relevance_link_flair_text = clean_text(flair_raw, use_demoji=True) if flair_raw else ""
            relevance_title = clean_text(title_raw, blocked_texts=TITLE_BLOCKED_TEXTS, use_demoji=True) if title_raw else ""
            relevance_selftext = (
                clean_text(selftext_raw, blocked_texts=SELFTEXT_BLOCKED_TEXTS, use_demoji=True)
                if selftext_raw
                else ""
            )

            relevance_parts = [f"Subreddit: {submission.get('subreddit')}"]
            if relevance_link_flair_text:
                relevance_parts.append(f"Link flair text: {relevance_link_flair_text}")
            if relevance_title:
                relevance_parts.append(f"Title: {relevance_title}")
            if relevance_selftext:
                relevance_parts.append(f"Selftext: {relevance_selftext}")

            valid_texts.append("\n".join(relevance_parts))
            valid_indices.append(idx)

    if valid_texts:
        try:
            for chunk_idx in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[chunk_idx : chunk_idx + batch_size]
                batch_indices = valid_indices[chunk_idx : chunk_idx + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)[:, 1]
                for predict_idx, prob_tensor in enumerate(probs):
                    results[batch_indices[predict_idx]] = 1 if prob_tensor.item() >= threshold else 0
        except Exception as exc:
            logger.error(f"Relevance prediction error: {exc}")

    return results


BOT_PATTERN = re.compile(
    r"""(?ix)\b(
        i\s*am\s*a\s*bot
        | i(?:['’])?m\s*a\s*bot
        | beep(?:[\s\.,!?-]*)boop
        | this\s+action\s+was\s+performed\s+automatically
        | downvote\s+to\s+(?:remove|delete)
        | send\s+a\s+modmail
        | contact\s+(?:the\s+)?(?:mod|mods|moderator|moderators)
    )\b"""
)
BOT_AUTHOR_SUBMISSIONS = {
    "AboriginalHelper",
    "AdAdministrative5611",
    "AdministrativeElk238",
    "AdministrativeGas633",
    "Akirapie",
    "Alert-Variation-6339",
    "AlertWoodpecker77",
    "Alert_Contest_2083",
    "alfred_bot2",
    "AndrewBot88",
    "AnyInformation7662",
    "apexproinfo1",
    "Aromatic-Bottle-9099",
    "AuthorBoth971",
    "Automatic-Alps5637",
    "Automatic-Holiday509",
    "Automatic-Selection2",
    "AutomaticInside7500",
    "Automatic_Mortgage76",
    "Automatic_Quality_45",
    "Automatic_Rock9090",
    "AutoModerator",
    "bakerydoge_mod",
    "baltotokenofficial",
    "beebottech",
    "Beneficial_Feed3729",
    "Best-Description921",
    "Blackboy-Feedback188",
    "Both_Yak601",
    "Capital-Ad-7665",
    "Capital-Signal9358",
    "Capital-Slice-4604",
    "Capital-Solid6750",
    "CapitalBarber9336",
    "CapitalGuava6966",
    "Capital_Ad9574",
    "Capital_Attention770",
    "CointestAdmin",
    "Conscious_Feedback_8",
    "cryptocalbot",
    "CRYPTOMOONOFFICIAL",
    "crypto_bot",
    "Crypto_Bot12",
    "cvcbot",
    "DryTip2644",
    "elComodoro",
    "Equivalent-Mode2249",
    "Financial_Tip_9221",
    "Flashy-Tip-8605",
    "Forex_CapitalM1",
    "Grand-Feedback",
    "GrowlingRapidity",
    "HuapadOfficial",
    "IcyAdministration677",
    "Individual-Monitor32",
    "Informal-Isopod6450",
    "Informal-Smoke-1475",
    "InformationTerrible8",
    "Interesting-Tip-1810",
    "jasonkurwapierdole",
    "julyteamod2",
    "kamodo2006",
    "Kapisaur",
    "Lcashofficial",
    "LogAdministrative478",
    "MarkTheRobot",
    "Modelae",
    "modis57544",
    "MoneyMakingMachine69",
    "moonpumpofficial",
    "Narrow_Tip_8356",
    "officialairdriop",
    "official_codenaft",
    "PancakeSniperBotV2",
    "papichuloxq",
    "pofapi",
    "Pure-Description3796",
    "Puzzleheaded_Tip7944",
    "rBitcoinMod",
    "rbtc-tipper",
    "RealBot123",
    "robograndpa",
    "Sea-Description-3858",
    "SeaworthinessBoth556",
    "SHIBALOKIMOD",
    "somebot12",
    "SpaceShrek-Official",
    "SwordfishCapital",
    "TapInternational3",
    "TaskAdministrative42",
    "TonaNova",
    "Valuable-Capital8063",
    "Wild-Mode-9930",
    "_Kapital-patates_",
}
BOT_AUTHOR_COMMENTS = {
    "AlexaPlayBot",
    "alphabet_order_bot",
    "AmputatorBot",
    "anti-gif-bot",
    "Anti-ThisBot-IB",
    "auddbot",
    "AutoModerator",
    "autotldr",
    "Banano_Tipbot",
    "BinanceCrypto-ModTeam",
    "birth-day-bot",
    "BitcoinBrasil-ModTeam",
    "Bitcoincash-ModTeam",
    "bitcoincashSV-ModTeam",
    "BitcoinMarkets-ModTeam",
    "BitcoinMining-ModTeam",
    "BitsTipper",
    "Bitty_Bot",
    "bot-sleuth-bot",
    "BsvAlertBot",
    "ccModBot",
    "chaintip",
    "CM-ModBot",
    "coinfeeds-bot",
    "CointestAdmin",
    "CointestMod",
    "comfort_bot_1962",
    "CommunityCurrencyBot",
    "CryptoContextModBot",
    "CryptoContextModBot2",
    "CryptoCurrency-ModTeam",
    "CryptoHelp-ModTeam",
    "CryptoMarkets-ModTeam",
    "CryptoMods",
    "CryptoTechnology-ModTeam",
    "Crypto_com-ModTeam",
    "DACapitalTrading",
    "Daytrading-ModTeam",
    "ectbot",
    "EncouragementRobot",
    "FatFingerHelperBot",
    "FeedMyTummy",
    "floodassistant",
    "FloodgatesBot",
    "freebanbot_lion",
    "freebanbot_spider",
    "freebanbot_squirrel",
    "Generic_Reddit_Bot",
    "gonano4",
    "Grammar-Bot-Elite",
    "grlctipsbot",
    "haikusbot",
    "IamYodaBot",
    "ioWxss6_bot",
    "JustAnAlpacaBot",
    "LearnDifferenceBot",
    "lerobinbot",
    "LinkifyBot",
    "lntipbot",
    "LuckyNumber-Bot",
    "MAGIC_EYE_BOT",
    "ModToolBot",
    "MoonsModBot",
    "moons_bot",
    "Nano10dollarDateBot",
    "NanoPredictionBot",
    "nano_tipper",
    "nano_tips",
    "nice___bot",
    "NoGoogleAMPBot",
    "of_patrol_bot",
    "pepetipbot",
    "qui_bot",
    "rBitcoinMod",
    "rbtc-tipper",
    "Reddit-Book-Bot",
    "remindditbot",
    "RemindMeBot",
    "reply-guy-bot",
    "RepostSleuthBot",
    "reputatorbot",
    "snappycoinbot",
    "sneakpeekbot",
    "SokkaHaikuBot",
    "SpambotSwatter",
    "the_timezone_bot",
    "ThisIsARepostBotBot",
    "timee_bot",
    "tiny_smile_bot",
    "twitterInfo_bot",
    "UkraineWithoutTheBot",
    "useles-converter-bot",
    "WallStreetBetsCrypto-ModTeam",
    "WaterIsWetBot",
    "WikiMobileLinkBot",
    "wikipedia_text_bot",
    "WikiSummarizerBot",
    "WikiTextBot",
    "WSBCryptoBot",
    "XGramatik-Bot",
    "XGramatikInsights-ModTeam",
    "YoMommaJokeBot",
    "_MoonBot",
}
def detect_submissions_bot(submissions):
    """Rule-based bot detection for submissions."""
    if not submissions:
        return []

    results = [None] * len(submissions)
    for idx, submission in enumerate(submissions):
        if submission.get("relevance") == 1:
            author = str(submission.get("author", ""))
            bot_title = (
                clean_text(
                    submission.get("title", ""),
                    blocked_texts=TITLE_BLOCKED_TEXTS,
                    remove_tokens=True,
                    lower=True,
                )
                if submission.get("title")
                else ""
            )
            bot_selftext = (
                clean_text(
                    submission.get("selftext", ""),
                    blocked_texts=SELFTEXT_BLOCKED_TEXTS,
                    remove_tokens=True,
                    lower=True,
                )
                if submission.get("selftext")
                else ""
            )
            text = " ".join(part for part in [bot_title, bot_selftext] if part)
            results[idx] = 1 if author in BOT_AUTHOR_SUBMISSIONS or bool(BOT_PATTERN.search(text)) else 0
    return results
def detect_comments_bot(comments):
    """Rule-based bot detection for comments."""
    if not comments:
        return []

    results = [None] * len(comments)
    for idx, comment in enumerate(comments):
        if comment.get("relevance") == 1:
            author = str(comment.get("author", ""))
            text = (
                clean_text(
                    comment.get("body", ""),
                    blocked_texts=BODY_BLOCKED_TEXTS,
                    remove_tokens=True,
                    lower=True,
                )
                if comment.get("body")
                else ""
            )
            results[idx] = 1 if author in BOT_AUTHOR_COMMENTS or bool(BOT_PATTERN.search(text)) else 0
    return results


def detect_submissions_sentiment(
    submissions,
    tokenizer_1,
    tokenizer_2,
    model_1,
    model_2,
    device,
    max_length=256,
    batch_size=16,
):
    """Predict submission sentiment with a weighted two-model ensemble."""
    if not submissions:
        return []

    valid_texts = []
    valid_indices = []
    results = [None] * len(submissions)
    w_roberta = torch.tensor([0.58, 0.60, 0.38], device=device)
    w_crypto = torch.tensor([0.42, 0.40, 0.62], device=device)

    for idx, submission in enumerate(submissions):
        if submission.get("bot") == 0:
            title = submission.get("title")
            selftext = submission.get("selftext")
            sentiment_title = clean_text(title, blocked_texts=TITLE_BLOCKED_TEXTS) if title else ""
            sentiment_selftext = clean_text(selftext, blocked_texts=SELFTEXT_BLOCKED_TEXTS) if selftext else ""
            sentiment_parts = []
            if sentiment_title:
                sentiment_parts.append(f"Title: {sentiment_title}")
            if sentiment_selftext:
                sentiment_parts.append(f"Selftext: {sentiment_selftext}")
            text_sentiment = "\n".join(sentiment_parts).strip()
            if text_sentiment:
                valid_texts.append(text_sentiment)
                valid_indices.append(idx)

    if valid_texts:
        try:
            for chunk_idx in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[chunk_idx : chunk_idx + batch_size]
                batch_indices = valid_indices[chunk_idx : chunk_idx + batch_size]
                inputs_1 = tokenizer_1(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs_2 = tokenizer_2(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs_1 = {key: value.to(device) for key, value in inputs_1.items()}
                inputs_2 = {key: value.to(device) for key, value in inputs_2.items()}
                with torch.no_grad():
                    logits_1 = model_1(**inputs_1).logits
                    probs_1 = torch.softmax(logits_1, dim=1)
                    logits_2 = model_2(**inputs_2).logits
                    probs_2 = torch.softmax(logits_2, dim=1)
                    probs_ensemble = (probs_1 * w_roberta) + (probs_2 * w_crypto)
                    probs_ensemble = probs_ensemble / probs_ensemble.sum(dim=1, keepdim=True)
                    preds = torch.argmax(probs_ensemble, dim=1)
                for predict_idx, pred_tensor in enumerate(preds):
                    results[batch_indices[predict_idx]] = pred_tensor.item()
        except Exception as exc:
            logger.error(f"Sentiment prediction error (submissions): {exc}")

    return results


def detect_comments_sentiment(
    comments,
    tokenizer_1,
    tokenizer_2,
    model_1,
    model_2,
    device,
    max_length=256,
    batch_size=32,
):
    """Predict comment sentiment with a weighted two-model ensemble."""
    if not comments:
        return []

    valid_texts = []
    valid_indices = []
    results = [None] * len(comments)
    w_roberta = torch.tensor([0.58, 0.60, 0.38], device=device)
    w_crypto = torch.tensor([0.42, 0.40, 0.62], device=device)

    for idx, comment in enumerate(comments):
        if comment.get("bot") == 0:
            body = comment.get("body")
            text_sentiment = clean_text(body, blocked_texts=BODY_BLOCKED_TEXTS) if body else ""
            if text_sentiment:
                valid_texts.append(text_sentiment)
                valid_indices.append(idx)

    if valid_texts:
        try:
            for chunk_idx in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[chunk_idx : chunk_idx + batch_size]
                batch_indices = valid_indices[chunk_idx : chunk_idx + batch_size]
                inputs_1 = tokenizer_1(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs_2 = tokenizer_2(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs_1 = {key: value.to(device) for key, value in inputs_1.items()}
                inputs_2 = {key: value.to(device) for key, value in inputs_2.items()}
                with torch.no_grad():
                    logits_1 = model_1(**inputs_1).logits
                    probs_1 = torch.softmax(logits_1, dim=1)
                    logits_2 = model_2(**inputs_2).logits
                    probs_2 = torch.softmax(logits_2, dim=1)
                    probs_ensemble = (probs_1 * w_roberta) + (probs_2 * w_crypto)
                    probs_ensemble = probs_ensemble / probs_ensemble.sum(dim=1, keepdim=True)
                    preds = torch.argmax(probs_ensemble, dim=1)
                for predict_idx, pred_tensor in enumerate(preds):
                    results[batch_indices[predict_idx]] = pred_tensor.item()
        except Exception as exc:
            logger.error(f"Sentiment prediction error (comments): {exc}")

    return results


BETA = 0.075
EPSILON = 0.55
def is_active(list_active_time, epsilon, beta, ref_time):
    if len(list_active_time) == 0:
        return False
    times = np.array(list_active_time, dtype=float)
    times_norm = (times - times.min()) / 3600
    ref_time_norm = (ref_time - times.min()) / 3600
    span = times_norm.max()
    if span > 7 * 24: return False
    s = np.sum(np.exp(-beta * (ref_time_norm - times_norm)))
    return False if s < epsilon else True


async def fetch_submission_task(client=None, subreddit=None, watchlist_subreddit=None, watchlist_submission=None, cookie_manager = None, first_run=None, stop_time=None, semaphore=None):
    """
    Worker xử lý cào cuốn chiếu độc lập cho riêng TỪNG subreddit.
    Nhiệm vụ: Cào đến khi nào hết bài viết mới thì thôi.
    """
    async with semaphore:
        sub_submissions = []
        success = True
        known_submission_ids = set(watchlist_submission.keys())
        if first_run or not watchlist_subreddit[subreddit].get("latest_submission_id"):
            temp = await fetch_submissions(
                    client=client,
                    subreddit=subreddit,
                    limit=100,
                    after=None,
                    before=None,
                    count=None,
                    show="all",
                    sr_detail=False,
                    user_agent=get_user_agent(),
                    cookie=cookie_manager.get_cookie(),
                )
            if temp is not None and len(temp) > 0:
                watchlist_subreddit[subreddit]["latest_submission_id"] = max(temp, key=lambda x: x["created_utc"])["id"]
            while True:
                if temp is None:
                    success = False  
                    break          
                if len(temp) == 0:
                    break  
                if any(item["id"] in known_submission_ids for item in temp) or any(item["created_utc"] < stop_time for item in temp):
                    sub_submissions.extend([
                        item for item in temp
                        if item["id"] not in known_submission_ids
                        and item["created_utc"] >= stop_time
                    ])
                    break
                sub_submissions.extend(temp)
                after_id = min(temp, key=lambda x: x["created_utc"])["id"]
                temp = await fetch_submissions(
                    client=client,
                    subreddit=subreddit,
                    limit=100,
                    after=f"t3_{after_id}",
                    before=None,
                    count=None,
                    show="all",
                    sr_detail=False,
                    user_agent=get_user_agent(),
                    cookie=cookie_manager.get_cookie(),                
                )
            if success:
                watchlist_subreddit[subreddit]["checked"] = True
            logger.info(
                f"success={success}, "
                f"type=after, "
                f"fetched={len(sub_submissions)}, "
                f"subreddit={subreddit}"
            )
            return sub_submissions

        else:
            before_id = watchlist_subreddit[subreddit].get("latest_submission_id")
            temp = await fetch_submissions(
                    client=client,
                    subreddit=subreddit,
                    limit=100,
                    after=None,
                    before=f"t3_{before_id}",
                    count=None,
                    show="all",
                    sr_detail=False,
                    user_agent=get_user_agent(),
                    cookie=cookie_manager.get_cookie(),
                )            
            while True:
                if temp is None:
                    success = False  
                    break          
                if len(temp) == 0:
                    break 
                watchlist_subreddit[subreddit]["latest_submission_id"] = max(temp, key=lambda x: x["created_utc"])["id"]
                if any(item["id"] in known_submission_ids for item in temp) or any(item["created_utc"] < stop_time for item in temp):
                    sub_submissions.extend([
                        item for item in temp
                        if item["id"] not in known_submission_ids
                        and item["created_utc"] >= stop_time
                    ])
                    break 
                sub_submissions.extend(temp)
                before_id = watchlist_subreddit[subreddit].get("latest_submission_id")
                temp = await fetch_submissions(
                    client=client,
                    subreddit=subreddit,
                    limit=100,
                    after=None,
                    before=f"t3_{before_id}",
                    count=None,
                    show="all",
                    sr_detail=False,
                    user_agent=get_user_agent(),
                    cookie=cookie_manager.get_cookie(),
                )
            if success:
                watchlist_subreddit[subreddit]["checked"] = True
            logger.info(
                f"success={success}, "
                f"type=before, "
                f"fetched={len(sub_submissions)}, "
                f"subreddit={subreddit}"
            )
            return sub_submissions


async def fetch_submission_pipeline(client, watchlist_subreddit, watchlist_submission, cookie_manager, first_run):
    """Run submission fetchers in parallel for all unchecked subreddits."""
    remaining_subreddits = sum(1 for info in watchlist_subreddit.values() if not info["checked"])
    logger.info(f"[reddit] submissions fetch start: subreddits={remaining_subreddits}")
    stop_time = datetime.now(timezone.utc).timestamp() - (30 * 24 * 60 * 60)
    semaphore = asyncio.Semaphore(16)
    tasks = [
        fetch_submission_task(
            client,
            subreddit=sub,
            watchlist_subreddit=watchlist_subreddit,
            watchlist_submission=watchlist_submission,
            cookie_manager=cookie_manager,
            first_run=first_run,
            stop_time=stop_time,
            semaphore=semaphore
        )
        for sub, info in watchlist_subreddit.items()
        if not info["checked"]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    submissions = []
    for sub_list in results:
        if isinstance(sub_list, list):
            submissions.extend(sub_list)
    logger.info(f"[reddit] submissions fetch done: {len(submissions)}")
    return submissions


REDDIT_SUBMISSIONS_TOPIC = "reddit-submissions"
def process_submissions_pipeline(
    submissions,
    watchlist_submission,
    model_language,
    tokenizer_relevance,
    model_relevance,
    threshold,
    tokenizer_sentiment_1,
    tokenizer_sentiment_2,
    model_1_sentiment,
    model_2_sentiment,
    device,
    producer=None,
    topic=REDDIT_SUBMISSIONS_TOPIC,
    batch_size=16,
):
    """Run submission enrichment pipeline and update the in-memory watchlist."""
    logger.info("[reddit] submissions process start")
    if not submissions:
        logger.info("[reddit] submissions process done")
        return

    logger.info("[reddit] submissions process step: language")
    languages = detect_submissions_language(submissions, model_language)
    for sub, lang in zip(submissions, languages):
        sub["language"] = lang

    logger.info("[reddit] submissions process step: relevance")
    relevances = detect_submissions_relevance(
        submissions=submissions,
        tokenizer=tokenizer_relevance,
        model=model_relevance,
        threshold=threshold,
        device=device,
        batch_size=batch_size,
    )
    for sub, rel in zip(submissions, relevances):
        sub["relevance"] = rel

    logger.info("[reddit] submissions process step: bot")
    bots = detect_submissions_bot(submissions)
    for sub, bot in zip(submissions, bots):
        sub["bot"] = bot

    logger.info("[reddit] submissions process step: sentiment")
    sentiments = detect_submissions_sentiment(
        submissions=submissions,
        tokenizer_1=tokenizer_sentiment_1,
        tokenizer_2=tokenizer_sentiment_2,
        model_1=model_1_sentiment,
        model_2=model_2_sentiment,
        device=device,
        batch_size=batch_size,
    )
    for sub, sent in zip(submissions, sentiments):
        sub["sentiment"] = sent

    logger.info("[reddit] submissions process step: transform")
    normalized_submissions = [transform_submission_record(sub) for sub in submissions]

    logger.info("[reddit] submissions process step: watchlist")
    for sub in normalized_submissions:
        sub_id = sub["id"]
        watchlist_submission[sub_id] = {
            "active": True,
            "list_active_time": [sub["created_utc"]],
            "list_comment_id": [],
            "relevance": sub["relevance"],
            "checked": False,
        }

    submissions[:] = normalized_submissions
    logger.info("[reddit] submissions process step: publish")
    publish_records(normalized_submissions, producer=producer, topic=topic, key_field="id")
    logger.info("[reddit] submissions process done")


async def fetch_comment_task(client, submission_id, watchlist_submission, cookie_manager = None, semaphore=None):
    """
    Worker xử lý cào comment bất đồng bộ cho riêng TỪNG bài viết (submission).
    """
    async with semaphore:
        comments = []
        success = True
        
        # --- NHÁNH 1: BÀI VIẾT MỚI TINH (Chưa có comment id nào trong bộ nhớ) ---
        if not watchlist_submission[submission_id]["list_comment_id"]:
            temp, more_children_ids = await fetch_comments(
                client=client,
                submission_id=submission_id,
                depth=100,
                sort="new",
                user_agent=get_user_agent(),
                cookie=cookie_manager.get_cookie(),
                
                limit=100,
                showedits = False,
                showmedia = False,
                showmore = True,
                showtitle = False,  
                threaded = False,  
                sr_detail=False,   
            )
            if temp is not None and more_children_ids is not None:
                comments.extend(temp)
                if len(more_children_ids) > 0:
                    for i in range(0, len(more_children_ids), 32):
                        chunk_ids = more_children_ids[i : i + 32]
                        temp_more = await fetch_comments(
                            client=client,
                            submission_id=submission_id,
                            depth=100,
                            sort="new",
                            user_agent=get_user_agent(),
                            cookie=cookie_manager.get_cookie(),
                            
                            children_ids=chunk_ids,
                            limit_children=False,
                        )
                        if temp_more is None:
                            success = False  
                            break
                        comments_dict = {c["id"]: c for c in (comments + temp_more) if c and "id" in c}
                        comments = list(comments_dict.values())
            else:
                sucess = False
        # --- NHÁNH 2: BÀI VIẾT CŨ (Cào tăng dần limit để săn comment mới) ---
        else:
            i = 8
            old_ids_set = set(watchlist_submission[submission_id]["list_comment_id"])
            while True:
                temp, _ = await fetch_comments(
                    client=client,
                    submission_id=submission_id,
                    depth=100,
                    sort="new",
                    user_agent=get_user_agent(),
                    cookie=cookie_manager.get_cookie(),
                    
                    limit=i+1,
                    showedits = False,
                    showmedia = False,
                    showmore = True,
                    showtitle = False,  
                    threaded = False,  
                    sr_detail=False,   
                )
                if temp is None:
                    success = False 
                    break
                if len(temp)==0:
                    break
                if any(item and item.get("id") in old_ids_set for item in temp) or i > 256:
                    comments.extend([item for item in temp if item and item.get("id") not in old_ids_set])
                    break
                else:
                    i = i * 2  
        if success:
            watchlist_submission[submission_id]["checked"] = True
        logger.info(
            f"success={success}, "
            f"submission={submission_id}, "
            f"fetched={len(comments)}"
        )
        
        return comments


async def fetch_comment_pipeline(client, watchlist_submission, cookie_manager):
    submissions_to_crawl = [
        s_id for s_id, info in watchlist_submission.items()
        if not info["checked"] and info["active"]
    ]
    logger.info(f"[reddit] comments fetch start: submissions={len(submissions_to_crawl)}")
    semaphore = asyncio.Semaphore(64)
    tasks = [fetch_comment_task(
            client=client,
            submission_id=s_id,
            watchlist_submission=watchlist_submission,
            cookie_manager=cookie_manager,
            semaphore=semaphore,)
        for s_id in submissions_to_crawl]
    results = await asyncio.gather(*tasks)
    all_comments = []
    for comment_list in results:
        all_comments.extend(comment_list)
    logger.info(f"[reddit] comments fetch done: {len(all_comments)}")
    return all_comments


REDDIT_COMMENTS_TOPIC = "reddit-comments"
def process_comments_pipeline(
    comments,
    watchlist_submission,
    model_language,
    tokenizer_sentiment_1,
    tokenizer_sentiment_2,
    model_1_sentiment,
    model_2_sentiment,
    device,
    producer=None,
    topic=REDDIT_COMMENTS_TOPIC,
    batch_size=16,
):
    """Run comment enrichment pipeline and update submission activity state."""
    logger.info("[reddit] comments process start")
    if not comments:
        logger.info("[reddit] comments process done")
        return

    logger.info("[reddit] comments process step: language")
    languages = detect_comments_language(comments, model_language)
    for comment, lang in zip(comments, languages):
        comment["language"] = lang

    logger.info("[reddit] comments process step: relevance")
    for comment in comments:
        sub_id = comment["link_id"].replace("t3_", "")
        comment["relevance"] = None
        comment["bot"] = None
        comment["sentiment"] = None

        is_english = comment["language"] == "en"
        is_emoji_only = comment["language"] == "empty" and emoji.emoji_count(comment.get("body", "")) > 0
        if (is_english or is_emoji_only) and sub_id in watchlist_submission:
            comment["relevance"] = watchlist_submission[sub_id]["relevance"]

    logger.info("[reddit] comments process step: bot")
    bots = detect_comments_bot(comments)
    for comment, bot in zip(comments, bots):
        comment["bot"] = bot

    logger.info("[reddit] comments process step: sentiment")
    sentiments = detect_comments_sentiment(
        comments=comments,
        tokenizer_1=tokenizer_sentiment_1,
        tokenizer_2=tokenizer_sentiment_2,
        model_1=model_1_sentiment,
        model_2=model_2_sentiment,
        device=device,
        batch_size=batch_size,
    )
    for comment, sent in zip(comments, sentiments):
        comment["sentiment"] = sent

    logger.info("[reddit] comments process step: transform")
    normalized_comments = [transform_comment_record(comment) for comment in comments]

    logger.info("[reddit] comments process step: watchlist")
    for comment in normalized_comments:
        sub_id = comment["link_id"].replace("t3_", "")
        if sub_id in watchlist_submission:
            watchlist_submission[sub_id]["list_comment_id"].append(comment["id"])
            watchlist_submission[sub_id]["list_active_time"].append(comment["created_utc"])

    comments[:] = normalized_comments
    logger.info("[reddit] comments process step: publish")
    publish_records(normalized_comments, producer=producer, topic=topic, key_field="id")
    logger.info("[reddit] comments process done")

def run_tracking_pipeline(
    client,
    watchlist_subreddit,
    watchlist_submission,
    cookie_manager,
    first_run,
    model_language,
    tokenizer_relevance,
    model_relevance,
    threshold,
    tokenizer_sentiment_1,
    tokenizer_sentiment_2,
    model_sentiment_1,
    model_sentiment_2,
    device,
    batch_size,
    submission_producer=None,
    comment_producer=None,
):
    logger.info(f"[reddit] run tracking start: first_run={first_run}")
    for sub_name in watchlist_subreddit.keys():
        watchlist_subreddit[sub_name]["checked"] = False
    for sub_id in watchlist_submission.keys():
        watchlist_submission[sub_id]["checked"] = False

    is_sub_remained = True
    is_post_remained = True
    loop_idx = 0
 
    while (is_sub_remained or is_post_remained) and cookie_manager.status() > 0: 
        loop_idx += 1
        logger.info(f"[reddit] run tracking loop: {loop_idx}")
        submissions = asyncio.run(fetch_submission_pipeline(client, watchlist_subreddit, watchlist_submission, cookie_manager, first_run))    
        process_submissions_pipeline(
            submissions=submissions,
            watchlist_submission=watchlist_submission,
            model_language=model_language,
            tokenizer_relevance=tokenizer_relevance,
            model_relevance=model_relevance,
            threshold=threshold,
            tokenizer_sentiment_1=tokenizer_sentiment_1,
            tokenizer_sentiment_2=tokenizer_sentiment_2,
            model_1_sentiment=model_sentiment_1,
            model_2_sentiment=model_sentiment_2,
            device=device,
            producer=submission_producer,
            batch_size=batch_size,
        )
        del submissions
        comments = asyncio.run(fetch_comment_pipeline(client, watchlist_submission, cookie_manager))
        process_comments_pipeline(
            comments=comments,
            watchlist_submission=watchlist_submission,
            model_language=model_language,
            tokenizer_sentiment_1=tokenizer_sentiment_1,
            tokenizer_sentiment_2=tokenizer_sentiment_2,
            model_1_sentiment=model_sentiment_1,
            model_2_sentiment=model_sentiment_2,
            device=device,
            producer=comment_producer,
            batch_size=batch_size,
        )
        del comments
        now_utc = int(datetime.now(timezone.utc).timestamp())

        for sub_id in list(watchlist_submission.keys()):
            list_active_time = watchlist_submission[sub_id]["list_active_time"]
            if watchlist_submission[sub_id]["active"]:
                watchlist_submission[sub_id]["active"]=is_active(list_active_time, EPSILON, BETA, now_utc)    
            if not watchlist_submission[sub_id]["active"]:
                oldest_active_time = min(list_active_time)
                if (now_utc - oldest_active_time) > (30 * 24 * 60 * 60):
                    del watchlist_submission[sub_id]

        is_sub_remained = any(
            not info.get("checked", False)
            for info in watchlist_subreddit.values()
        )

        is_post_remained = any(
            (not info.get("checked", False)) and info.get("active", False)
            for info in watchlist_submission.values()
        )
    active_submission_count = sum(
        1 for info in watchlist_submission.values() if info.get("active", False)
    )
    logger.info(f"[reddit] run tracking done: watchlist_submission={active_submission_count}")
