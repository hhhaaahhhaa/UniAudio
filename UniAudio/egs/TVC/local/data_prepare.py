import os
from argparse import ArgumentParser, Namespace


def main(args):
    src, dst = args.src, args.dst
    for subset in ["train", "validation", "test"]:
        root = f"{src}/{subset}"
        os.makedirs(f"{dst}/{subset}", exist_ok=True)
        source_wav_scp_path = f"{dst}/{subset}/source_wav.scp"
        target_wav_scp_path = f"{dst}/{subset}/wav.scp"
        text_path = f"{dst}/{subset}/text"
        instruction_path = f"{dst}/{subset}/instruction.scp"

        ids = []
        for path in os.listdir(f"{root}/source"):
            id = path[:-4]
            ids.append(id)
        
        with open(source_wav_scp_path, 'w') as f:
            for id in ids:
                f.write(f"{id} {root}/source/{id}.wav\n")
        with open(target_wav_scp_path, 'w') as f:
            for id in ids:
                f.write(f"{id} {root}/target/{id}.wav\n")
        with open(text_path, 'w') as f:
            for id in ids:
                with open(f"{root}/transcription/{id}.txt", "r") as f1:
                    txt = f1.read()
                f.write(f"{id} {txt}\n")
        with open(instruction_path, 'w') as f:
            for id in ids:
                with open(f"{root}/instruction/{id}.txt", "r") as f1:
                    txt = f1.read()
                f.write(f"{id} {txt}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("dst", type=str)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
