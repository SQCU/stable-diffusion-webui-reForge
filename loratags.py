import json
import re

from argparse import ArgumentParser
from pathlib import Path

NONESCPAREN_REGEX = re.compile(r'(?<!\\)[()]')

def parse_args():
    parser = ArgumentParser(description="get a prompt from a lora's built-in tags")
    parser.add_argument("lorafile", type=Path, help="path to the lora file in safetensors format")
    parser.add_argument("-countmin", type=int, default=10, help="minimum tag count for a tag to be included")
    parser.add_argument("-tagmax", type=int, default=30, help="maximum amount of tags to include in the prompt")
    return parser.parse_args()

def read_metadata_from_safetensors(filename) -> dict:
    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == "{":
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass
        return res

def main():
    args = parse_args()
    lorafile = args.lorafile.resolve()
    if not lorafile.exists():
        print(f"'{lorafile.name}' does not exist")
        exit(1)
    if not lorafile.is_file():
        print(f"'{lorafile.name}' is not a file")
        exit(1)

    try:
        metadata = read_metadata_from_safetensors(lorafile)
    except Exception as e:
        print(e)
        exit(1)

    tag_frequency: dict = metadata.get("ss_tag_frequency")
    if tag_frequency is None:
        print("lora has no tag metadata.")
        exit(1)

    # In the case of multiple groups, join all their tags together
    all_tags = {}
    for tags in tag_frequency.values():
        tags: dict
        if not all_tags:
            all_tags = tags.copy()
        else:
            for tag, count in tags.items():
                try:
                    all_tags[tag] += count
                except KeyError:
                    all_tags[tag] = count

    tags = sorted(all_tags.items(), key=lambda item: item[1], reverse=True)
    tags = list(filter(lambda item: item[1] >= args.countmin, tags))
    tags = [NONESCPAREN_REGEX.sub(r'\\\g<0>', item[0]).strip() for item in tags[0:args.tagmax]]

    print(', '.join(tags))


if __name__ == "__main__":
    main()
