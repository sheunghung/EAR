import json
import torch
import os
import glob
import argparse
from tqdm import tqdm
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

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

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto"
    )
    model.eval()

    text_prompt_template = "Please answer the question based on the audio. Question: {QUESTION}"

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
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt_text}
                ]}
            ]

            try:
                text_input = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )

                audios, images, videos = process_mm_info(
                    conversation,
                    use_audio_in_video=False
                )

                inputs = processor(
                    text=text_input,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=False
                )

                inputs = inputs.to(model.device).to(model.dtype)

                eos_token_ids = [
                    processor.tokenizer.eos_token_id,
                    processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
                ]

                with torch.inference_mode():
                    generate_ids = model.generate(
                        **inputs,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=eos_token_ids
                    )

                generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
                response = processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                sample["model_response"] = response

            except Exception as e:
                sample["model_response"] = f"[Error: {str(e)}]"
                torch.cuda.empty_cache()

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f_out.flush()

if __name__ == "__main__":
    main()