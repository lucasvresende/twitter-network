from pathlib import Path

TWITTER_NETWORK = Path(__file__).resolve().parent
REPO = TWITTER_NETWORK.parent.parent
DATA = REPO / "data"
DATA_OUT = DATA / "out"
TWITTER_COMBINED_TXT = DATA / "twitter_combined.txt"
