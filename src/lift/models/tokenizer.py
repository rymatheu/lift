#

from transformers import AutoTokenizer
import numpy as np


MODEL_ID = "principled-intelligence/gemma-4-E2B-it-text-only"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return tokenizer
