from datasets import load_dataset
from tqdm import tqdm
import json


def main():
    info = {}
    for k in ["train", "valid", "test"]:
        ds = load_dataset(
            "zion84006/tencentdata_speech_tokenizer",
            split=k,
            streaming=True,
        )
        if k == "valid":
            split_name = "validation"
        else:
            split_name = k
        info[split_name] = []
        for instance in tqdm(ds):
            data = {
                "file_id": instance["file_id"],
                "wav_id": instance["wav_id"],
                "instruction": instance["instruction"],
                "transcription": instance["transcription"]
            }
            info[split_name].append(data)
    with open("tvc_metadata.json", 'w') as f:
        json.dump(f, indent=4)


if __name__ == "__main__":
    main()
