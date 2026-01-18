import json
from tqdm import tqdm
from desta import DeSTA25AudioModel
import os
import argparse
import torch
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model = DeSTA25AudioModel.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    text_prompt_template = "Please answer the question based on the audio. Question: {QUESTION} \n Audio:\n<|AUDIO|>"

    dataset = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for sample in tqdm(dataset):
            sample_id = sample.get("id")
            pattern = os.path.join(args.audio_dir, f"{sample_id}*.wav")
            matches = glob.glob(pattern)

            if not matches:
                sample["model_response"] = f"[Error: No audio found for pattern {pattern}]"
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_out.flush()
                continue

            audio_path = matches[0]
            prompt_text = text_prompt_template.format(QUESTION=sample["question"])

            messages = [
                {"role": "system", "content": "Focus on the audio clips and instructions."},
                {
                    "role": "user",
                    "content": prompt_text,
                    "audios": [{"audio": audio_path, "text": None}]
                }
            ]

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        messages=messages,
                        do_sample=False,
                        top_p=1.0,
                        temperature=1.0,
                        max_new_tokens=256
                    )

                sample["model_response"] = outputs.text[0].strip()

            except Exception as e:
                sample["model_response"] = f"[Error: {str(e)}]"
                torch.cuda.empty_cache()

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f_out.flush()

if __name__ == "__main__":
    main()