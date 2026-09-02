import import_ipynb
from RSS_feedparser import summarize_and_classify
import argparse
import os
from dotenv import load_dotenv
import sys
import time

def news_classifier_cli(url:str="http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml", limit:int=2):
    return None
def safe_summarize_and_classify(client, title: str, text: str, categories: list[str])-> dict:
    try:
        summarize_and_classify(client, title, text, categories)
    except AuthenticationError as e:
        print(e)
        sys.exit
    except RateLimitError:
        time.sleep(10)
        summarize_and_classify(client, title, text, categories)
    else:
        
    return None
parser = argparse.ArgumentParser(prog="news_classifier_cli")
parser.add_argument("url")
parser.add_argument("limit")
load_dotenv()
api_key = os.get("OPENAI_API_KEY")

