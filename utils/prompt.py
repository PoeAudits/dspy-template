import dspy
import os
from pathlib import Path
import re

def get_prompt(optimized_program, adapter: dspy.ChatAdapter = dspy.ChatAdapter()):
    prompt = { 
        name: adapter.format(
            p.signature,
            demos=p.demos,
            inputs={k: f"{{{k}}}" for k in p.signature.input_fields},
            )
            for name, p in optimized_program.named_predictors()
            }['self']
    return prompt

def with_instructions_from_file(signature: type[dspy.Signature], prompt_file: str, re_exp: str = r"[ \t]+$") -> type[dspy.Signature]:
    prompt_path = Path(prompt_file)

    if not prompt_path.exists():
        raise NotADirectoryError(f"Prompt file not found: {prompt_file}")

    with prompt_path.open("r", encoding="utf-8") as f:
        content = f.read()
        clean_text = re.sub(re_exp, "", content, flags=re.MULTILINE)
        clean_text = re.sub(r'[ \t]+$', '', clean_text, flags=re.M)
        clean_text = re.sub(r'\n\s*\n+', '\n\n', clean_text.strip())
        return signature.with_instructions(clean_text)

def save_program(signature: type[dspy.Signature], lm: dspy.LM, prompt_path: str, program_path: str):
        program = dspy.Predict(with_instructions_from_file(signature, prompt_path))
        program.set_lm(lm)
        os.makedirs(program_path, exist_ok=True)
        program.save(path=program_path, save_program=True)
