import hashlib
import os
import chardet
import tiktoken
import anthropic

from vertexai.preview import tokenization

def calculate_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
        return hash_md5.hexdigest()


def list_files(path, include_ignore_file=False, add_path=False, extension=None):
    filenames = os.listdir(path)
    if not include_ignore_file:
        filenames = [f for f in filenames if not f.startswith('.')]
    if extension is not None:
        filenames = [f for f in filenames if f.endswith(extension)]
    if add_path:
        filenames = [os.path.join(path, f) for f in filenames]
    return filenames


def detect_encoding(filepath):
    with open(filepath, 'rb') as f:
        raw_data = f.read()
        enc = chardet.detect(raw_data)
        return enc['encoding']


def count_tokens(text, model):
    if model == "gpt":
        encoding = tiktoken.encoding_for_model('gpt-4')
        enc = encoding.encode(text)
        return len(enc)
    elif model == "claude":
        ac = anthropic.Client()
        return ac.count_tokens(text)
    elif model == "gemini":
        model_name = "gemini-1.5-pro-001"
        tokenizer = tokenization.get_tokenizer_for_model(model_name)
        result = tokenizer.count_tokens(text)
        return result.total_tokens
