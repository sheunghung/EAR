import json
import torch
import os
import argparse
import re
from tqdm import tqdm
from transformers import (
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--audio_root_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(args.input_file)

    if not os.path.isdir(args.audio_root_dir):
        raise NotADirectoryError(args.audio_root_dir)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True
    )

    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()

    dataset = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for sample in tqdm(dataset):

            question = sample.get("question", "")
            sample_id = str(sample.get("id"))

            pattern = re.compile(
                rf"^{re.escape(sample_id)}(?:\D.*)?\.wav$"
            )

            audio_files = [
                os.path.join(args.audio_root_dir, fname)
                for fname in os.listdir(args.audio_root_dir)
                if pattern.match(fname)
            ]

            audio_files = sorted(audio_files)
            audio_path = audio_files[0]

            prompt = f"Please answer the question based on the audio.\n{question}"
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": audio_path},
                    ],
                }
            ]

            try:
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )

                inputs = inputs.to(model.device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )

                generated_ids = outputs[:, inputs.input_ids.shape[1]:]
                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                sample["model_response"] = response

            except Exception as e:
                sample["model_response"] = f"[Error: {str(e)}]"
                torch.cuda.empty_cache()

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()