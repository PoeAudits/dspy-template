import dspy
from pathlib import Path
from typing import Optional
import re

def with_instructions_from_file(signature: type[dspy.Signature], prompt_file: str, re_exp: Optional[str] = r"[ \t]+$") -> type[dspy.Signature]:
    prompt_path = Path(prompt_file)

    if not prompt_path.exists():
        raise NotADirectoryError(f"Prompt file not found: {prompt_file}")

    with prompt_path.open("r", encoding="utf-8") as f:
        content = f.read()
        clean = re.sub(re_exp, "", content, flags=re.MULTILINE)
        return signature.with_instructions(clean)

