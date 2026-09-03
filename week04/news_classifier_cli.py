import import_ipynb
import os
from dotenv import load_dotenv
load_dotenv()
from RSS_feedparser import summarize_and_classify
from RSS_feedparser import fetch_articles
import argparse


import sys
import time
import openai


parser = argparse.ArgumentParser(prog="news_classifier_cli")
parser.add_argument("--url")
parser.add_argument("--limit",type=int)
args = parser.parse_args()
url = args.url
limit = args.limit


def news_classifier_cli(url:str="http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml", limit:int=2):
    result = []
    client = openai.OpenAI()
    articles = fetch_articles(url, limit)
    for article in articles:
        result.append(safe_summarize_and_classify(client, article.get("title"), article.get("summary"), ["Technology", "Science", "Health", "Entertainment"]))
    return result


def safe_summarize_and_classify(client, title: str, text: str, categories: list[str], retries=3)-> dict:
    result = None
    attempts = 1
    try:
        result = summarize_and_classify(client, title, text, categories)
    except openai.AuthenticationError as e:
        print(e)
        sys.exit(1)
    except openai.RateLimitError:
        if retries > 0:
            for attempt in range(attempts):
                time.sleep(2**attempt)
                attempt += 1
                result = safe_summarize_and_classify(client, title, text, categories,retries-1)
        else:
            print("failed - try later")
    except openai.APIError:
        result = {"title": title, "summary": None, "category": "Error"}

    return result
result = news_classifier_cli(url, limit)

import json
from datetime import datetime
point = datetime.now().isoformat()
results = {"timestamp":point, "results":result}
with open(f"articles_{datetime.now().strftime("%Y%m%d_%H%M%S")}", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)