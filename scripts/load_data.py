import json


def load_json_text(json_file: str, tok_num: int = 500):
    with open(json_file, "r") as f:
        json_dict = json.load(f)
    abstract_sec = json_dict.get("abstract")
    if abstract_sec:
        abstract = abstract_sec.get("abstract", "")
    else:
        abstract = ""
    content = []
    for section in json_dict["sections"]:
        text = section.get("text", "")
        content.append(text)
    all_text = " ".join([abstract] + content)
    return all_text.split()[:tok_num]


if __name__ == '__main__':
    pass
