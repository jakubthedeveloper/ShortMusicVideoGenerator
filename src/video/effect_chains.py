import random
from video.effects import VIDEO_EFFECTS

def random_chain(min_len=2, max_len=4):
    """Returns a random list of effects"""
    chain_len = random.randint(min_len, max_len)
    return random.sample(VIDEO_EFFECTS, chain_len)

