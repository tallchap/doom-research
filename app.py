import asyncio
import os
import re
import json
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import requests as http_requests


def _get_git_version():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


GIT_VERSION = _get_git_version()

try:
    import redis
except Exception:
    redis = None

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)

# Use env override first. Fallback list tries likely-available Anthropic models.
MODEL = os.getenv("ANTHROPIC_MODEL", "")
FALLBACK_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
]
WEB_SEARCH_TOOL_TYPES = ["web_search_20260209", "web_search_20250305"]
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# GPT Researcher config
GPTR_ENABLED = os.getenv("GPTR_ENABLED", "true").lower() == "true"
GPTR_LLM_PROVIDER = os.getenv("GPTR_LLM_PROVIDER", "openai")  # openai, anthropic, etc.
GPTR_FAST_LLM = os.getenv("GPTR_FAST_LLM", "openai:gpt-4o-mini")
GPTR_SMART_LLM = os.getenv("GPTR_SMART_LLM", "openai:gpt-5.4")
GPTR_STRATEGIC_LLM = os.getenv("GPTR_STRATEGIC_LLM", "openai:o3")
GPTR_REPORT_TYPE = os.getenv("GPTR_REPORT_TYPE", "research_report")
GPTR_MAX_SEARCH_RESULTS = int(os.getenv("GPTR_MAX_SEARCH_RESULTS", "20"))
GPTR_MIN_SOURCES = int(os.getenv("GPTR_MIN_SOURCES", "8"))
GPTR_MAX_PASSES = int(os.getenv("GPTR_MAX_PASSES", "2"))
GPTR_RETRIEVER = os.getenv("GPTR_RETRIEVER", "tavily")

# Map GPTR_* settings to the unprefixed env vars that GPT Researcher reads.
# Done once at startup to avoid repeated process-global mutations in threads.
if GPTR_ENABLED:
    os.environ["FAST_LLM"] = GPTR_FAST_LLM
    os.environ["SMART_LLM"] = GPTR_SMART_LLM
    os.environ["STRATEGIC_LLM"] = GPTR_STRATEGIC_LLM
    os.environ["MAX_SEARCH_RESULTS_PER_QUERY"] = str(GPTR_MAX_SEARCH_RESULTS)
    os.environ["RETRIEVER"] = GPTR_RETRIEVER

REDIS_URL = os.getenv("REDIS_URL", "").strip()
RESEARCH_QUEUE_NAME = os.getenv("RESEARCH_QUEUE_NAME", "research_jobs")
RESEARCH_USE_WORKER = os.getenv("RESEARCH_USE_WORKER", "true").lower() == "true"
RESEARCH_JOB_KEY_PREFIX = os.getenv("RESEARCH_JOB_KEY_PREFIX", "research:job:")

executor = ThreadPoolExecutor(max_workers=16)
research_lock = threading.Lock()


@dataclass
class ResearchJobState:
    job_id: str
    created_at: float
    mode: str = "standard"
    status: str = "queued"
    logs: List[str] = field(default_factory=list)
    report: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    tool_used: Optional[str] = None
    request_payload: Dict = field(default_factory=dict)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0


RESEARCH_JOBS: Dict[str, ResearchJobState] = {}

RESEARCH_JOBS_DIR = os.getenv("RESEARCH_JOBS_DIR", "/tmp/render-research-jobs")
os.makedirs(RESEARCH_JOBS_DIR, exist_ok=True)

def _redis_client():
    if not REDIS_URL or redis is None:
        return None
    try:
        import ssl as _ssl
        kwargs = {"decode_responses": True}
        if REDIS_URL.startswith("rediss://"):
            kwargs["ssl_cert_reqs"] = _ssl.CERT_NONE
        return redis.from_url(REDIS_URL, **kwargs)
    except Exception:
        return None


def _redis_job_key(job_id: str) -> str:
    return f"{RESEARCH_JOB_KEY_PREFIX}{job_id}"


def enqueue_research_job(job_id: str) -> bool:
    r = _redis_client()
    if not r:
        print(f"[enqueue] Redis client unavailable for job {job_id}")
        return False
    try:
        r.lpush(RESEARCH_QUEUE_NAME, job_id)
        print(f"[enqueue] Job {job_id} queued to Redis")
        return True
    except Exception as e:
        print(f"[enqueue] Redis lpush failed for job {job_id}: {e}")
        return False


def _job_file_path(job_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", job_id)
    return os.path.join(RESEARCH_JOBS_DIR, f"{safe_id}.json")


def persist_research_job(job: ResearchJobState) -> None:
    payload = asdict(job)
    try:
        path = _job_file_path(job.job_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass

    r = _redis_client()
    if r:
        try:
            r.set(_redis_job_key(job.job_id), json.dumps(payload, ensure_ascii=False), ex=86400)
        except Exception:
            pass


def _safe_build_research_job(data: dict) -> ResearchJobState:
    known = {f.name for f in fields(ResearchJobState)}
    return ResearchJobState(**{k: v for k, v in data.items() if k in known})


def load_research_job(job_id: str) -> Optional[ResearchJobState]:
    r = _redis_client()
    if r:
        try:
            raw = r.get(_redis_job_key(job_id))
            if raw:
                return _safe_build_research_job(json.loads(raw))
        except Exception:
            pass

    try:
        path = _job_file_path(job_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _safe_build_research_job(data)
    except Exception:
        return None


def log_research(job: ResearchJobState, message: str):
    ts = time.strftime("%H:%M:%S")
    job.logs.append(f"[{ts}] {message}")
    persist_research_job(job)


def build_research_prompt(contact_name: str, target: str, research_prompt: str, deep_research: bool = False) -> str:
    target_section = f"\nTarget:\n{target}" if target.strip() else ""
    if deep_research:
        return f"""
You are running a deep research pass for outbound personalization.

Research Contact:
{contact_name}{target_section}
Research Request:
{research_prompt}

Requirements:
- Prioritize verifiable information with explicit evidence.
- If sources conflict, call out the conflict.
- Distinguish facts vs inference.
- If uncertain, say "unknown" instead of guessing.

Output in this exact structure:
1) Executive Snapshot (5-8 bullets)
2) Evidence Matrix (Claim | Source URL | Date Published | Evidence | Confidence | Relevance)
3) Contradictions / Ambiguities
4) Outreach Angles
5) Unknowns / Next Verification Steps

IMPORTANT: All citations must include the publication date.
Use format: (Author/Source, YYYY-MM-DD) or (Source, Month YYYY).
In the Evidence Matrix, include a 'Date Published' column.
In the References section, include full dates for every entry.
""".strip()

    return f"""
You are a research assistant helping prepare personalized outreach.

Research Contact:
{contact_name}
{target_section}
Research Request:
{research_prompt}

Output requirements:
- Keep it factual, concise, and useful for sales/outreach prep.
- Use this structure:
  1) Snapshot
  2) Relevant background
  3) Current priorities / likely pain points
  4) Messaging angles
  5) Risks / unknowns / what to verify
- If data is uncertain, say so explicitly.
""".strip()


# ── YouTube API ──
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
YT_TRANSCRIPT_SERVICE = "https://youtube-transcript-data.replit.app/transcript"

# ── Twitter API (ported from local-pitch-helper) ──
TWITTERAPI_KEY = os.getenv("TWITTERAPI_KEY", "")
TWITTER_BASE_URL = "https://api.twitterapi.io"
_twitter_last_request = 0


def _twitter_throttle():
    """Free tier: 1 request per 5 seconds."""
    global _twitter_last_request
    elapsed = time.time() - _twitter_last_request
    if elapsed < 5.5:
        time.sleep(5.5 - elapsed)
    _twitter_last_request = time.time()


def twitter_get_user_profile(username: str) -> dict:
    _twitter_throttle()
    r = http_requests.get(
        f"{TWITTER_BASE_URL}/twitter/user/info",
        headers={"X-API-Key": TWITTERAPI_KEY},
        params={"userName": username},
    )
    r.raise_for_status()
    return r.json().get("data", {})


def twitter_get_recent_tweets(username: str, count: int = 20) -> list:
    tweets = []
    cursor = ""
    while len(tweets) < count:
        _twitter_throttle()
        r = http_requests.get(
            f"{TWITTER_BASE_URL}/twitter/user/last_tweets",
            headers={"X-API-Key": TWITTERAPI_KEY},
            params={"userName": username, "includeReplies": "false", "cursor": cursor},
        )
        r.raise_for_status()
        data = r.json()
        batch = data.get("tweets", [])
        if not batch:
            break
        tweets.extend(batch)
        if not data.get("has_next_page"):
            break
        cursor = data.get("next_cursor", "")
    return tweets[:count]


def twitter_format_profile(profile: dict) -> str:
    bio = profile.get("profile_bio", {}).get("description", profile.get("description", ""))
    return (
        f"**{profile.get('name')}** (@{profile.get('userName')})\n"
        f"Bio: {bio}\n"
        f"Followers: {profile.get('followers', 0):,} | Following: {profile.get('following', 0):,}\n"
        f"Tweets: {profile.get('statusesCount', 0):,} | Location: {profile.get('location', 'N/A')}\n"
        f"Joined: {profile.get('createdAt', 'N/A')}\n"
        f"DMs open: {profile.get('canDm', False)}"
    )


def twitter_format_tweets(tweets: list) -> str:
    lines = []
    for t in tweets:
        lines.append(
            f"{t['createdAt']}\n"
            f"LIKES: {t.get('likeCount', 0):,}  VIEWS: {t.get('viewCount', 0):,}  "
            f"RTs: {t.get('retweetCount', 0):,}\n"
            f"{t['text'][:400]}\n"
            f"{t['url']}\n---"
        )
    return "\n".join(lines)


def twitter_search_tweets(query: str, query_type: str = "Latest", max_pages: int = 1) -> list:
    """Advanced tweet search via TwitterAPI.io."""
    tweets = []
    cursor = ""
    for _ in range(max_pages):
        _twitter_throttle()
        r = http_requests.get(
            f"{TWITTER_BASE_URL}/twitter/tweet/advanced_search",
            headers={"X-API-Key": TWITTERAPI_KEY},
            params={"query": query, "queryType": query_type, "cursor": cursor},
        )
        r.raise_for_status()
        data = r.json()
        batch = data.get("tweets", [])
        tweets.extend(batch)
        if not data.get("has_next_page"):
            break
        cursor = data.get("next_cursor", "")
    return tweets


def _search_twitter_handle(contact_name: str) -> Optional[str]:
    """Search for a person's Twitter handle by name using tweet search."""
    if not TWITTERAPI_KEY:
        return None
    try:
        # Search for tweets mentioning/from this person
        tweets = twitter_search_tweets(contact_name, query_type="Top", max_pages=1)
        if not tweets:
            return None
        # Count usernames — the person's own account often appears most
        from collections import Counter
        usernames = Counter()
        name_lower = contact_name.lower().replace(" ", "")
        for t in tweets:
            author = t.get("author", {})
            uname = author.get("userName", "")
            display = (author.get("name", "") or "").lower().replace(" ", "")
            followers = author.get("followers", 0) or 0
            # Boost score if display name closely matches contact name
            score = 1
            if name_lower in display or display in name_lower:
                score = 100 + followers
            usernames[uname] += score
        if usernames:
            best = usernames.most_common(1)[0][0]
            return best
    except Exception as e:
        print(f"[twitter] Handle search failed for '{contact_name}': {e}")
    return None


def _fetch_twitter_context(contact_name: str, job, topic: str = "") -> str:
    """Auto-search Twitter handle, fetch profile + tweets + topic search, return formatted context."""
    if not TWITTERAPI_KEY:
        return ""
    try:
        log_research(job, f"Twitter: searching for handle for '{contact_name}'...")
        handle = _search_twitter_handle(contact_name)
        if not handle:
            log_research(job, "Twitter: no handle found, skipping")
            return ""
        log_research(job, f"Twitter: found @{handle}, fetching profile + tweets...")
        profile = twitter_get_user_profile(handle)
        tweets = twitter_get_recent_tweets(handle, count=20)
        result = twitter_format_profile(profile)
        if tweets:
            result += "\n\n## Recent Tweets\n" + twitter_format_tweets(tweets)
        log_research(job, f"Twitter: fetched profile + {len(tweets)} tweets for @{handle}")

        # Also search for topic-relevant tweets about/by this person
        if topic:
            try:
                # Search for tweets FROM the person about the topic, plus tweets mentioning them
                from_query = f"from:{handle} {topic}" if handle else f"{contact_name} {topic}"
                log_research(job, f"Twitter: searching topic tweets: '{from_query[:60]}'...")
                topic_tweets = twitter_search_tweets(from_query, query_type="Top", max_pages=1)
                # Also search mentions if from: query returns few results
                if len(topic_tweets) < 3 and handle:
                    mention_query = f"@{handle} {topic}"
                    mention_tweets = twitter_search_tweets(mention_query, query_type="Top", max_pages=1)
                    topic_tweets.extend(mention_tweets)
                # Deduplicate against already-fetched tweets
                existing_ids = {t.get("id") or t.get("tweetId") for t in tweets} if tweets else set()
                new_tweets = [t for t in topic_tweets if (t.get("id") or t.get("tweetId")) not in existing_ids]
                if new_tweets:
                    result += f"\n\n## Topic-Relevant Tweets ({topic[:40]})\n" + twitter_format_tweets(new_tweets[:10])
                    log_research(job, f"Twitter: found {len(new_tweets)} topic-relevant tweets")
                else:
                    log_research(job, "Twitter: no additional topic-relevant tweets found")
            except Exception as e:
                log_research(job, f"Twitter: topic search failed (non-fatal): {e}")

        return result
    except Exception as e:
        log_research(job, f"Twitter: enrichment failed: {e}")
        return ""


# ── YouTube enrichment ──

def _yt_format_duration(iso_duration: str) -> str:
    """Parse ISO 8601 duration (PT1H2M3S) to human-readable."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration or "")
    if not m:
        return iso_duration or "Unknown"
    h, mins, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    if h:
        return f"{h}:{mins:02d}:{s:02d}"
    return f"{mins}:{s:02d}"


def _yt_duration_seconds(iso_duration: str) -> int:
    """Parse ISO 8601 duration to total seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration or "")
    if not m:
        return 0
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)


def _search_youtube_videos(query: str, max_results: int = 5) -> List[dict]:
    """Search YouTube Data API v3 for long videos (>10min)."""
    if not YOUTUBE_API_KEY:
        return []
    try:
        # Step 1: Search
        r = http_requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "q": query,
                "key": YOUTUBE_API_KEY,
                "part": "snippet",
                "type": "video",
                "videoDuration": "long",  # >20min
                "maxResults": max_results,
                "order": "relevance",
            },
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return []

        # Step 2: Get video details for duration/stats
        video_ids = ",".join(item["id"]["videoId"] for item in items)
        r2 = http_requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "id": video_ids,
                "key": YOUTUBE_API_KEY,
                "part": "snippet,contentDetails,statistics",
            },
        )
        r2.raise_for_status()
        details = {v["id"]: v for v in r2.json().get("items", [])}

        results = []
        for item in items:
            vid = item["id"]["videoId"]
            detail = details.get(vid, {})
            snippet = detail.get("snippet", item.get("snippet", {}))
            content = detail.get("contentDetails", {})
            stats = detail.get("statistics", {})
            results.append({
                "video_id": vid,
                "title": snippet.get("title", ""),
                "published_at": snippet.get("publishedAt", ""),
                "channel": snippet.get("channelTitle", ""),
                "duration": _yt_format_duration(content.get("duration", "")),
                "duration_seconds": _yt_duration_seconds(content.get("duration", "")),
                "view_count": int(stats.get("viewCount", 0)),
            })
        return results
    except Exception as e:
        print(f"[youtube] Search failed: {e}")
        return []


def _get_youtube_transcript(video_id: str, max_chars: int = 15000, keywords: list = None) -> Optional[str]:
    """Fetch transcript, return keyword-relevant sections (not just the beginning)."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        r = http_requests.post(YT_TRANSCRIPT_SERVICE, json={"url": url}, timeout=30)
        r.raise_for_status()
        data = r.json()
        transcript_data = data.get("transcript_data", [])
        if not transcript_data:
            return None
        # Build timestamped lines
        lines = []
        for entry in transcript_data:
            ts = entry.get("start", 0)
            mins, secs = divmod(int(ts), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                ts_str = f"[{hours}:{mins:02d}:{secs:02d}]"
            else:
                ts_str = f"[{mins:02d}:{secs:02d}]"
            lines.append(f"{ts_str} {entry.get('text', '')}")

        full_text = "\n".join(lines)
        # If it fits, return everything
        if len(full_text) <= max_chars:
            return full_text

        # Keyword-focused extraction: score windows of 10 lines
        if keywords:
            kw_lower = [k.lower() for k in keywords if len(k) > 1]
            window_size = 10
            windows = []
            for i in range(0, len(lines), window_size):
                chunk_lines = lines[i:i + window_size]
                chunk_text = "\n".join(chunk_lines)
                score = sum(1 for kw in kw_lower if kw in chunk_text.lower())
                windows.append((i, score, chunk_text))

            # Take top-scoring windows first
            scored = [w for w in windows if w[1] > 0]
            if scored:
                scored.sort(key=lambda w: -w[1])
                selected = []
                budget = max_chars
                for idx, score, text in scored:
                    if len(text) + 10 > budget:
                        break
                    selected.append((idx, text))
                    budget -= len(text) + 10
                # Re-sort by position in transcript
                selected.sort(key=lambda w: w[0])
                parts = []
                prev_idx = -999
                for idx, text in selected:
                    if idx > prev_idx + 10:
                        parts.append("[...]")
                    parts.append(text)
                    prev_idx = idx
                result = "\n".join(parts)
                print(f"[youtube] Keyword extraction: {len(scored)} relevant windows from {len(windows)} total for {video_id}")
                return result

        # Fallback: evenly sample beginning, middle, end
        n = len(lines)
        sample_size = n // 5  # 20% each
        beginning = lines[:sample_size]
        mid_start = (n - sample_size) // 2
        middle = lines[mid_start:mid_start + sample_size]
        end = lines[n - sample_size:]
        sampled = "\n".join(beginning) + "\n[...]\n" + "\n".join(middle) + "\n[...]\n" + "\n".join(end)
        if len(sampled) > max_chars:
            sampled = sampled[:max_chars] + "\n[...transcript truncated...]"
        return sampled
    except Exception as e:
        print(f"[youtube] Transcript fetch failed for {video_id}: {e}")
        return None


def _fetch_youtube_context(contact_name: str, topic: str, job, max_videos: int = 3) -> str:
    """Search YouTube for contact + topic, fetch transcripts, format for research."""
    if not YOUTUBE_API_KEY:
        return ""
    try:
        search_query = f"{contact_name} {topic} interview OR talk OR podcast"
        log_research(job, f"YouTube: searching for '{search_query[:80]}'...")
        videos = _search_youtube_videos(search_query, max_results=5)
        if not videos:
            log_research(job, "YouTube: no videos found, skipping")
            return ""
        log_research(job, f"YouTube: found {len(videos)} videos, fetching transcripts...")

        # Extract keywords from topic + contact name for focused transcript extraction
        stop_words = {'find', 'their', 'about', 'what', 'them', 'they', 'this', 'that', 'with', 'from', 'have', 'been', 'will', 'would', 'could', 'should', 'does', 'into', 'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out'}
        keywords = [w for w in topic.lower().split() if len(w) > 1 and w not in stop_words]
        # Include contact name parts to find their speaking segments
        name_parts = [p.lower() for p in contact_name.split() if len(p) > 1]
        keywords = list(dict.fromkeys(keywords + name_parts))  # dedupe, preserve order
        log_research(job, f"YouTube: using keywords {keywords[:10]} for transcript extraction")

        parts = []
        transcribed = 0
        for v in videos:
            if transcribed >= max_videos:
                break
            transcript = _get_youtube_transcript(v["video_id"], max_chars=15000, keywords=keywords)
            if not transcript:
                log_research(job, f"YouTube: no transcript for '{v['title'][:60]}'")
                continue
            transcribed += 1
            log_research(job, f"YouTube: transcribed '{v['title'][:60]}' ({len(transcript)} chars)")
            parts.append(
                f"### {v['title']} ({v['published_at'][:10]})\n"
                f"Channel: {v['channel']} | Duration: {v['duration']} | Views: {v['view_count']:,}\n"
                f"URL: https://youtube.com/watch?v={v['video_id']}\n\n"
                f"{transcript}"
            )

        if not parts:
            log_research(job, "YouTube: no transcripts available for any video")
            return ""
        return "\n\n".join(parts)
    except Exception as e:
        log_research(job, f"YouTube: enrichment failed: {e}")
        return ""


# ── Claude API helpers ──

def get_model_candidates() -> List[str]:
    models = []
    if MODEL.strip():
        models.append(MODEL.strip())
    models.extend(FALLBACK_MODELS)
    seen = set()
    ordered = []
    for m in models:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _extract_text_from_message(msg) -> str:
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _extract_usage(msg) -> Dict[str, int]:
    usage = getattr(msg, "usage", None)
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }


def generate_with_claude(prompt: str, deep_research: bool = False) -> Tuple[str, str, Optional[str], Dict[str, int]]:
    if not API_KEY:
        return (
            "No live model call ran because ANTHROPIC_API_KEY is missing.",
            "local-fallback-no-api-key",
            None,
            {"input_tokens": 0, "output_tokens": 0},
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=API_KEY, timeout=120.0)

    candidates = get_model_candidates()

    last_err = None
    for model_name in candidates:
        tool_types = [None]
        if deep_research:
            tool_types = WEB_SEARCH_TOOL_TYPES + [None]

        for tool_type in tool_types:
            try:
                kwargs = {
                    "model": model_name,
                    "max_tokens": 1500 if deep_research else 900,
                    "temperature": 0.5 if deep_research else 0.7,
                    "messages": [{"role": "user", "content": prompt}],
                }

                if tool_type:
                    kwargs["tools"] = [
                        {
                            "type": tool_type,
                            "name": "web_search",
                            "max_uses": 8,
                        }
                    ]

                print(f"[claude] Calling {model_name} (max_tokens={kwargs['max_tokens']}, tool={tool_type})...")
                msg = client.messages.create(**kwargs)
                usage = _extract_usage(msg)
                print(f"[claude] {model_name} responded (in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)})")
                return _extract_text_from_message(msg), model_name, tool_type, usage
            except Exception as e:
                err_text = str(e)
                last_err = e
                print(f"[claude] {model_name} error: {err_text[:200]}")

                if "not_found_error" in err_text or "model:" in err_text:
                    break

                if deep_research and tool_type and (
                    "web_search" in err_text
                    or "tool" in err_text.lower()
                    or "invalid_request_error" in err_text
                ):
                    continue

                raise

    raise RuntimeError(
        "No available Anthropic model/tool combo matched this API key. "
        f"Tried models: {', '.join(get_model_candidates())}. Last error: {last_err}"
    )


# ── GPT Researcher integration ──

class _GptrLogsHandler:
    """Custom logs handler that bridges GPT Researcher progress into ResearchJobState logs."""

    def __init__(self, job_id: str):
        self._job_id = job_id
        self.logs: List[Dict[str, Any]] = []

    async def send_json(self, data: Dict[str, Any]) -> None:
        self.logs.append(data)
        content = data.get("output", data.get("content", ""))
        if isinstance(content, str) and content.strip():
            msg = content.strip()[:4000]
        else:
            msg = json.dumps(data, default=str)[:4000]
        with research_lock:
            job = RESEARCH_JOBS.get(self._job_id)
            if job and len(job.logs) < 800:
                log_research(job, msg)


def _run_web_research(job, query, job_id):
    """Phase A: Run GPTR web research only (no social enrichment)."""
    from gpt_researcher import GPTResearcher

    log_research(job, "[web] Phase A started — GPTR web research")
    log_research(job, f"[web] Query: {query[:500]}")
    log_research(job, f"[web] LLM: {GPTR_LLM_PROVIDER} | Smart: {GPTR_SMART_LLM} | Fast: {GPTR_FAST_LLM}")

    logs_handler = _GptrLogsHandler(job_id)

    def _normalize_source(source):
        if isinstance(source, str):
            return source.strip()
        if isinstance(source, dict):
            return str(source.get("url") or source.get("source") or source.get("link") or "").strip()
        return str(source).strip()

    def _has_inline_citations(text):
        return bool(re.search(r"\[[0-9]+\]", text or "") or re.search(r"https?://", text or ""))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pass_reports = []
    all_sources = []
    total_costs = 0.0
    try:
        pass_count = max(1, min(4, GPTR_MAX_PASSES))
        for pass_idx in range(pass_count):
            pass_query = query
            if pass_idx > 0:
                pass_query = (
                    query
                    + "\n\nSecond pass requirements: expand coverage with additional independent sources, "
                    + "prioritize primary evidence, and include inline citations for factual claims."
                )

            log_research(job, f"[web] Research pass {pass_idx + 1}/{pass_count} started")
            researcher = GPTResearcher(
                query=pass_query,
                report_type=GPTR_REPORT_TYPE,
                report_source="online",
                tone="informative",
                websocket=logs_handler,
            )

            loop.run_until_complete(researcher.conduct_research())
            pass_report = loop.run_until_complete(researcher.write_report())
            pass_reports.append(pass_report or "")

            costs = researcher.get_costs()
            if isinstance(costs, (int, float)):
                total_costs += float(costs)

            raw_sources = researcher.get_research_sources() or []
            normalized = [s for s in (_normalize_source(x) for x in raw_sources) if s]
            for src in normalized:
                if src not in all_sources:
                    all_sources.append(src)

            # Fallback: extract URLs from report text when get_research_sources() returns empty
            if not all_sources and pass_reports[-1]:
                urls_in_report = re.findall(r'https?://[^\s\)\]>"]+', pass_reports[-1])
                for url in urls_in_report:
                    url = url.rstrip(".,;:")
                    if url not in all_sources:
                        all_sources.append(url)

            latest_report = pass_reports[-1]
            has_citations = _has_inline_citations(latest_report)
            log_research(job, f"[web] Pass {pass_idx + 1}: sources={len(all_sources)} citations={'yes' if has_citations else 'no'}")

            if len(all_sources) >= GPTR_MIN_SOURCES and has_citations:
                log_research(job, "[web] Quality gate met")
                break
            if pass_idx + 1 < pass_count:
                log_research(job, "[web] Quality gate not met; running another pass")
    finally:
        loop.close()

    web_report = (pass_reports[-1] if pass_reports else "") or ""
    log_research(job, f"[web] Phase A done — {len(all_sources)} sources, cost=${total_costs:.4f}")
    return web_report, all_sources, total_costs


def _run_social_enrichment(job, payload):
    """Phase B: Fetch Twitter + YouTube enrichment (runs in parallel with web research)."""
    log_research(job, "[social] Phase B started — Twitter + YouTube enrichment")
    parts = []

    # Twitter
    if TWITTERAPI_KEY:
        try:
            twitter_ctx = _fetch_twitter_context(
                payload['contact_name'], job, topic=payload.get('research_prompt', '')
            )
            if twitter_ctx:
                parts.append("## Twitter Profile & Recent Tweets\n" + twitter_ctx)
                log_research(job, f"[social] Twitter: collected {len(twitter_ctx)} chars")
            else:
                log_research(job, "[social] Twitter: returned empty")
        except Exception as e:
            log_research(job, f"[social] Twitter: crashed: {e}")
    else:
        log_research(job, "[social] Twitter: TWITTERAPI_KEY not set, skipping")

    # YouTube
    if YOUTUBE_API_KEY:
        try:
            yt_ctx = _fetch_youtube_context(
                payload['contact_name'], payload.get('research_prompt', ''), job, max_videos=3
            )
            if yt_ctx:
                parts.append("## YouTube Video Transcripts\n" + yt_ctx)
                log_research(job, f"[social] YouTube: collected {len(yt_ctx)} chars")
            else:
                log_research(job, "[social] YouTube: returned empty")
        except Exception as e:
            log_research(job, f"[social] YouTube: crashed: {e}")
    else:
        log_research(job, "[social] YouTube: YOUTUBE_API_KEY not set, skipping")

    social_report = "\n\n".join(parts) if parts else ""
    log_research(job, f"[social] Phase B done — {len(parts)} sources, {len(social_report)} chars")
    return social_report


def _synthesize_report(job, web_report, social_report, web_sources, subject, research_prompt):
    """Phase C: Use Claude to synthesize web + social research into a final report."""
    log_research(job, "[synthesis] Phase C started — consolidating web + social with Claude")

    # Build synthesis prompt
    prompt_parts = [
        f"You are a research analyst. Synthesize the following research data about {subject} into a single, "
        f"comprehensive, well-structured report.\n\n"
        f"The user's research request was: \"{research_prompt}\"\n\n"
    ]

    if web_report:
        prompt_parts.append(
            f"## WEB RESEARCH (from automated web search)\n\n{web_report}\n\n"
        )
    if social_report:
        prompt_parts.append(
            f"## SOCIAL & VIDEO SOURCES (Twitter profile/tweets + YouTube transcripts)\n\n{social_report}\n\n"
        )

    prompt_parts.append(
        "Produce a final structured report with these sections:\n"
        "1) **Executive Snapshot** (5-8 bullets summarizing key findings from ALL sources)\n"
        "2) **Evidence Matrix** (table: Claim | Source URL | Date Published | Evidence | Confidence | Relevance)\n"
        "3) **Contradictions / Ambiguities**\n"
        "4) **Outreach Angles** (practical approaches based on findings)\n"
        "5) **Unknowns / Next Verification Steps**\n"
        "6) **References** (all sources cited, with dates)\n\n"
        "IMPORTANT RULES:\n"
        "- Incorporate findings from BOTH web research AND social/video sources. Do not ignore either.\n"
        "- If Twitter or YouTube data reveals opinions, quotes, or positions, include them prominently.\n"
        "- All citations must include publication dates: (Source, YYYY-MM-DD).\n"
        "- Include direct quotes when available from tweets or video transcripts.\n"
        "- ALWAYS include clickable URLs for tweets (e.g. https://x.com/...) and YouTube videos (e.g. https://youtube.com/watch?v=...) when referencing them. Never cite a tweet or video without its URL.\n"
        "- In the Evidence Matrix, the Source URL column must contain the actual URL, not just a text description.\n"
        "- Be concrete and opinionated in your assessment — do not hedge excessively."
    )

    synthesis_prompt = "\n".join(prompt_parts)
    log_research(job, f"[synthesis] Prompt: {len(synthesis_prompt)} chars, calling Claude...")

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=API_KEY, timeout=180.0)
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

        msg = client.messages.create(
            model=model_name,
            max_tokens=8000,
            temperature=0.3,
            messages=[{"role": "user", "content": synthesis_prompt}],
        )
        usage = _extract_usage(msg)
        result = _extract_text_from_message(msg)
        log_research(job, f"[synthesis] Claude {model_name} responded ({usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out)")
        return result, model_name, usage
    except Exception as e:
        log_research(job, f"[synthesis] Claude synthesis failed: {e}")
        # Graceful degradation: concatenate reports without synthesis
        fallback = ""
        if web_report:
            fallback += "# Web Research\n\n" + web_report
        if social_report:
            fallback += "\n\n---\n# Social & Video Sources\n\n" + social_report
        log_research(job, "[synthesis] Falling back to concatenated report")
        return fallback or "No report generated.", "fallback", {"input_tokens": 0, "output_tokens": 0}


def _run_gptr_research(job_id: str) -> None:
    """Run parallel deep research: web (GPTR) + social (Twitter/YouTube), then Claude synthesis."""
    from concurrent.futures import ThreadPoolExecutor as _SynthPool

    with research_lock:
        job = RESEARCH_JOBS.get(job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = time.time()
        log_research(job, "Deep research started (parallel: web + social → synthesis)")

    try:
        payload = job.request_payload
        target = payload.get('target', '').strip()
        subject = f"{payload['contact_name']} at {target}" if target else payload['contact_name']
        query = (
            f"Research {subject}. "
            f"{payload['research_prompt']}\n\n"
            f"Produce a structured report with:\n"
            f"1) Executive Snapshot (5-8 bullets)\n"
            f"2) Evidence Matrix (Claim | Source URL | Date Published | Evidence | Confidence | Relevance)\n"
            f"3) Contradictions / Ambiguities\n"
            f"4) Outreach Angles\n"
            f"5) Unknowns / Next Verification Steps\n\n"
            f"IMPORTANT: All citations must include the publication date. "
            f"Use format: (Author/Source, YYYY-MM-DD) or (Source, Month YYYY). "
            f"In the Evidence Matrix, include a 'Date Published' column. "
            f"In the References section, include full dates for every entry."
        )

        # --- Phase A + B: run web research and social enrichment in parallel ---
        web_report = ""
        web_sources = []
        web_costs = 0.0
        social_report = ""

        with _SynthPool(max_workers=2) as pool:
            web_future = pool.submit(_run_web_research, job, query, job_id)
            social_future = pool.submit(_run_social_enrichment, job, payload)

            # Collect results with timeouts
            try:
                web_report, web_sources, web_costs = web_future.result(timeout=300)
            except Exception as e:
                log_research(job, f"[web] Phase A failed: {e}")

            try:
                social_report = social_future.result(timeout=120)
            except Exception as e:
                log_research(job, f"[social] Phase B failed: {e}")

        if not web_report and not social_report:
            raise RuntimeError("Both web and social research failed — no data to synthesize")

        # --- Phase C: Claude synthesis ---
        final_report, synth_model, synth_usage = _synthesize_report(
            job, web_report, social_report, web_sources,
            subject, payload.get('research_prompt', '')
        )

        # Append sources summary
        sources_summary = ""
        if web_sources:
            sources_summary = "\n\n---\n**Web Sources:**\n" + "\n".join(
                f"- {s}" for s in web_sources[:50]
            )

        full_report = final_report + sources_summary

        with research_lock:
            job = RESEARCH_JOBS[job_id]
            if job.status == "canceled":
                job.finished_at = time.time()
                return
            job.report = full_report[:60000]
            job.model_used = f"gpt-researcher ({GPTR_SMART_LLM}) + {synth_model}"
            job.tool_used = "gpt-researcher + claude-synthesis"
            job.status = "done"
            job.finished_at = time.time()
            job.input_tokens = synth_usage.get("input_tokens", 0)
            job.output_tokens = synth_usage.get("output_tokens", 0)
            log_research(job, f"Deep research completed. GPTR cost: ${web_costs:.4f}")
            if web_sources:
                log_research(job, f"Web sources: {len(web_sources)}")
    except Exception as e:
        with research_lock:
            job = RESEARCH_JOBS[job_id]
            job.status = "error"
            job.error = str(e)
            job.finished_at = time.time()
            log_research(job, f"Deep research failed: {e}")


def _run_api_research_job(job_id: str):
    with research_lock:
        job = RESEARCH_JOBS.get(job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = time.time()
        log_research(job, "Research job started")

    try:
        payload = job.request_payload
        deep_research = job.mode == "deep"
        if deep_research:
            log_research(job, "Deep research enabled (web search tool path)")

        prompt = build_research_prompt(
            contact_name=payload["contact_name"],
            target=payload["target"],
            research_prompt=payload["research_prompt"],
            deep_research=deep_research,
        )

        report, model_used, tool_used, usage = generate_with_claude(prompt, deep_research=deep_research)

        with research_lock:
            job = RESEARCH_JOBS[job_id]
            job.report = report
            job.model_used = model_used
            job.tool_used = tool_used
            job.status = "done"
            job.finished_at = time.time()
            job.input_tokens = usage.get("input_tokens", 0)
            job.output_tokens = usage.get("output_tokens", 0)
            if tool_used:
                log_research(job, f"Completed with model={model_used}, tool={tool_used}")
            else:
                log_research(job, f"Completed with model={model_used}")
    except Exception as e:
        with research_lock:
            job = RESEARCH_JOBS[job_id]
            job.status = "error"
            job.error = str(e)
            job.finished_at = time.time()
            log_research(job, f"Failed: {e}")


def run_research_job(job_id: str):
    with research_lock:
        job = RESEARCH_JOBS.get(job_id)
        if not job:
            return
        mode = job.mode

    if mode == "deep" and GPTR_ENABLED:
        _run_gptr_research(job_id)
    else:
        _run_api_research_job(job_id)


# ── Routes ──

@app.route("/")
def index():
    return render_template("index.html", git_version=GIT_VERSION)


@app.post("/research_contact")
def research_contact():
    data = request.get_json(force=True, silent=True) or {}

    required = ["contact_name", "research_prompt"]
    missing = [k for k in required if not str(data.get(k, "")).strip()]
    if missing:
        return jsonify({"ok": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    mode = str(data.get("mode", "standard")).strip().lower()
    if mode not in {"standard", "deep"}:
        return jsonify({"ok": False, "error": "Invalid mode. Use standard or deep"}), 400

    job_id = str(uuid.uuid4())
    job = ResearchJobState(
        job_id=job_id,
        created_at=time.time(),
        mode=mode,
        request_payload={
            "contact_name": data["contact_name"].strip(),
            "target": str(data.get("target", "")).strip(),
            "research_prompt": data["research_prompt"].strip(),
        },
    )

    log_research(job, f"Queued research job mode={mode}")

    with research_lock:
        RESEARCH_JOBS[job_id] = job

    persist_research_job(job)

    if RESEARCH_USE_WORKER and REDIS_URL:
        queued = enqueue_research_job(job_id)
        if not queued:
            return jsonify({"ok": False, "error": "Failed to enqueue research job"}), 500
    else:
        executor.submit(run_research_job, job_id)

    return jsonify({"ok": True, "job_id": job_id, "status": "queued", "mode": mode})


@app.get("/research_status")
def research_status():
    job_id = request.args.get("job_id", "")
    with research_lock:
        job = RESEARCH_JOBS.get(job_id)
        # When using Redis worker, always reload from Redis for fresh progress
        if job and RESEARCH_USE_WORKER and REDIS_URL and job.status not in ("done", "error", "timeout", "canceled"):
            fresh = load_research_job(job_id)
            if fresh:
                RESEARCH_JOBS[job_id] = fresh
                job = fresh
        if not job:
            disk_job = load_research_job(job_id)
            if disk_job:
                RESEARCH_JOBS[job_id] = disk_job
                job = disk_job
        if not job:
            return jsonify({"ok": False, "error": "job not found"}), 404

        return jsonify(
            {
                "ok": True,
                "job_id": job.job_id,
                "mode": job.mode,
                "status": job.status,
                "done": job.status in ["done", "error", "timeout", "canceled"],
                "error": job.error,
                "report": job.report,
                "model_used": job.model_used,
                "tool_used": job.tool_used,
                "logs": job.logs[-200:],
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "input_tokens": job.input_tokens,
                "output_tokens": job.output_tokens,
            }
        )


@app.post("/research_cancel")
def research_cancel():
    data = request.get_json(force=True, silent=True) or {}
    job_id = str(data.get("job_id", "")).strip()

    with research_lock:
        job = RESEARCH_JOBS.get(job_id)
        if not job:
            disk_job = load_research_job(job_id)
            if disk_job:
                RESEARCH_JOBS[job_id] = disk_job
                job = disk_job
        if not job:
            return jsonify({"ok": False, "error": "job not found"}), 404

        if job.status in ["done", "error", "timeout", "canceled"]:
            return jsonify({"ok": True, "status": job.status})

        job.status = "canceled"
        job.finished_at = time.time()
        log_research(job, "Cancel requested")

    return jsonify({"ok": True, "status": "canceled"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
