from pathlib import Path

TWITTER_NETWORK = Path(__file__).resolve().parent
REPO = TWITTER_NETWORK.parent.parent
DATA = REPO / "data"
DATA_OUT = DATA / "out"
DATA_TWITTER = DATA / "twitter"
TWITTER_COMBINED_TXT = DATA / "twitter_combined.txt"

FEATS = sorted(DATA_TWITTER.glob("*.feat"))
FEATNAMES = sorted(DATA_TWITTER.glob("*.featnames"))
