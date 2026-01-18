import sys
import os
import json
import ujson
import torch
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import *

sys.path.append(os.path.join(COSY_VOCODER))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def init_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model.training = False
    model.bind_processor(tokenizer, training=False, relative_path="/")

    return model, tokenizer


def construct_prompt(audio_path, question, system_prompt=""):
    audio_part = (
        audio_start_token +
        ujson.dumps({"path": audio_path}, ensure_ascii=False) +
        audio_end_token
    )

    text = ""
    if system_prompt:
        text += role_prefix["system"] + system_prompt

    text += role_prefix["user"]
    text += audio_part + "\n" + question
    text += role_prefix["assistant"]

    return text


def generate_text_only(model, tokenizer, content):
    pret = model.processor([content])

    out = model.generate(
        pret.input_ids.cuda(),
        attention_mask=pret.attention_mask.cuda(),
        audios=pret.audios.cuda() if pret.audios is not None else None,
        encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
        bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        stop_strings=["<|endoftext|>"],
        do_sample=True,
        temperature=0.8,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        return_dict_in_generate=True,
    )

    plen = pret.input_ids.shape[1]
    text = tokenizer.decode(out.sequences[0, plen:])
    return re.sub(r"<\|endoftext\|>", "", text).strip()


def main():
    args = parse_args()

    model, tokenizer = init_model(args.model_path)

    global audio_start_token, audio_end_token
    audio_start_token = tokenizer.convert_ids_to_tokens(
        model.config.audio_config.audio_start_token_id
    )
    audio_end_token = tokenizer.convert_ids_to_tokens(
        model.config.audio_config.audio_end_token_id
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in):
            if not line.strip():
                continue

            data = json.loads(line)
            sample_id = str(data["id"])
            question = data.get("question", "")

            audio_dir = args.audio_root
            if not os.path.isdir(audio_dir):
                data["model_response"] = None
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            pattern = re.compile(
                rf"^{re.escape(sample_id)}(?:\D.*)?\.wav$"
            )

            audio_files = [
                os.path.join(audio_dir, fname)
                for fname in os.listdir(audio_dir)
                if pattern.match(fname)
            ]

            if not audio_files:
                data["model_response"] = None
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            audio_path = sorted(audio_files)[0]

            try:
                prompt = construct_prompt(
                    audio_path,
                    question,
                    system_prompt="Please answer the question based on the audio."
                )
                data["model_response"] = generate_text_only(
                    model, tokenizer, prompt
                )
            except Exception:
                data["model_response"] = ""

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()