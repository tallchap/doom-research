import os
import time

from app import (
    RESEARCH_QUEUE_NAME,
    _redis_client,
    load_research_job,
    log_research,
    persist_research_job,
    run_research_job,
    RESEARCH_JOBS,
    research_lock,
)


def main():
    print("[worker] starting")
    while True:
        r = _redis_client()
        if not r:
            print("[worker] REDIS_URL unavailable; retrying in 5s")
            time.sleep(5)
            continue

        try:
            item = r.brpop(RESEARCH_QUEUE_NAME, timeout=10)
        except Exception as e:
            print(f"[worker] brpop error: {e}")
            time.sleep(2)
            continue

        if not item:
            continue

        _, job_id = item
        try:
            job = load_research_job(job_id)
            if not job:
                print(f"[worker] job not found: {job_id}")
                continue

            with research_lock:
                RESEARCH_JOBS[job_id] = job

            log_research(job, "Dequeued by worker")
            persist_research_job(job)
            run_research_job(job_id)
        except Exception as e:
            print(f"[worker] job {job_id} failed: {e}")


if __name__ == "__main__":
    main()
