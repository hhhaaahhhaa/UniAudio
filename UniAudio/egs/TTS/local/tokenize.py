from tqdm import tqdm

from tools.tokenizer.phone.text_tokenizer import Text2PhoneTokenizer


def main():
    data_root = "egs/TTS/data"
    ttokenizer = Text2PhoneTokenizer("checkpoints/lang_nosp")

    for name in ["train-clean-100", "dev-clean", "test-clean"]:
        with open(f"{data_root}/{name}/text", "r", encoding="utf-8") as f:
            res = []
            for line in tqdm(f):
                if line == "\n":
                    continue
                basename, txt = line.strip().split(" ", 1)
                res.append(f"{basename} {ttokenizer.tokenize(txt)}")
        with open(f"{data_root}/{name}/phone.scp", "w", encoding="utf-8") as f:
            for x in res:
                f.write(f"{x}\n")


if __name__ == "__main__":
    main()
