import os, sys

# ─── non-zero rank 프로세스의 stdout/stderr 를 /dev/null 로 리다이렉트 ───
# local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# if local_rank != 0:
#     sys.stdout = open(os.devnull, "w")
#     sys.stderr = open(os.devnull, "w")

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
