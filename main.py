from importlib.metadata import version
import importlib
import os
import urllib.request
import re

import tiktoken

from simple_tokenizer import SimpleTokenizerV1 

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))


if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


print("Total number of character:", len(raw_text))
print(raw_text[:99])

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]

print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

print(len(preprocessed))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])


# 建立分词字典，词： 编号
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))


# tokenizer = SimpleTokenizerV1(vocab)

# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."

# text = " <|endoftext|> ".join((text1, text2))

# print(text)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))

tokenizer = tiktoken.get_encoding("gpt2")

# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#      "of someunknownPlace."
# )

# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# print(integers)

# strings = tokenizer.decode(integers)

# print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))