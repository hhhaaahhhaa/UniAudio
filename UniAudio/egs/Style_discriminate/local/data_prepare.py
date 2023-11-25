import os
from argparse import ArgumentParser, Namespace
import random
from tqdm import tqdm


def parse_google(src: str, res):
    for subset in ["train", "validation", "test"]:
        root = f"{src}/{subset}"
        for path in tqdm(os.listdir(f"{root}/source")):
            id = path[:-4]
            with open(f"{root}/transcription/{id}.txt", "r") as f1:
                transcription = f1.read()
            with open(f"{root}/instruction/{id}.txt", "r") as f1:
                instruction = f1.read()
            data = {
                "id": id,
                "src": f"{root}/source/{id}.wav",
                "tgt": f"{root}/target/{id}.wav",
                "instruction": instruction,
                "transcription": transcription,
            }
            res[subset].append(data)


def parse_prompt_speech(src: str, res):
    for subset in ["train", "test"]:
        root = f"{src}/{subset}"
        for path in tqdm(os.listdir(f"{root}/source")):
            id = path[:-4]
            with open(f"{root}/transcription/{id}.txt", "r") as f1:
                transcription = f1.read()
            with open(f"{root}/instruction/{id}.txt", "r") as f1:
                instruction = f1.read()
            data = {
                "id": id,
                "src": f"{root}/source/{id}.wav",
                "tgt": f"{root}/target/{id}.wav",
                "instruction": instruction.strip(),
                "transcription": transcription.strip(),
            }
            res[subset].append(data)


def parse_tencent(src: str, res):
    import json
    with open(f"{src}/tvc_metadata.json", 'r') as f:
        info = json.load(f)
    for subset in info:
        for instance in info[subset]:
            id = instance["file_id"]
            wav_id = instance["wav_id"]
            data = {
                "id": id,
                "src": f"{src}/data/{wav_id}/source.wav",
                "tgt": f"{src}/data/{wav_id}/final_target.wav",
                "instruction": instance["instruction"],
                "transcription": instance["transcription"],
            }
            res[subset].append(data)


def main(args):
    dst = args.dst
    data_dict = {
        "train": [],
        "validation": [],
        "test": []
    }
    parse_google(args.src_google, data_dict)
    parse_prompt_speech(args.src_prompt_speech, data_dict)
    parse_tencent(args.src_tencent, data_dict)
    for subset in ["train", "validation", "test"]:
        os.makedirs(f"{dst}/{subset}", exist_ok=True)
        source_wav_scp_path = f"{dst}/{subset}/source_wav.scp"
        target_wav_scp_path = f"{dst}/{subset}/wav.scp"
        text_path = f"{dst}/{subset}/text"
        instruction_path = f"{dst}/{subset}/instruction.scp"
        label_path = f"{dst}/{subset}/label.scp"

        n = len(data_dict[subset])
        with open(source_wav_scp_path, 'w') as f:
            for i, sample in enumerate(data_dict[subset] * 2):
                f.write(f"{i:07d} {sample['src']}\n")
        with open(target_wav_scp_path, 'w') as f:
            for i, sample in enumerate(data_dict[subset] * 2):
                f.write(f"{i:07d} {sample['tgt']}\n")
        with open(text_path, 'w') as f:
            for i, sample in enumerate(data_dict[subset] * 2):
                f.write(f"{i:07d} {sample['transcription']}\n")
        with open(instruction_path, 'w') as f:
            for i, sample in enumerate(data_dict[subset]):
                f.write(f"{i:07d} {sample['instruction']}\n")
            for i in range(n):
                sample = random.sample(data_dict[subset], k=1)[0]
                f.write(f"{i + n:07d} {sample['instruction']}\n")
        with open(label_path, 'w') as f:
            for i in range(2 * n):
                if i < n:
                    f.write(f"{i:07d} 1\n")
                else:
                    f.write(f"{i:07d} 0\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("src_google", type=str)
    parser.add_argument("src_prompt_speech", type=str)
    parser.add_argument("src_tencent", type=str)
    parser.add_argument("dst", type=str)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    random.seed(666)
    args = parse_args()
    main(args)
