import dspy
import re
import inspect
import copy
from pathlib import Path
from typing import Optional, Type, Callable, List

class ProgramGenerator(dspy.Module):
    """Base class for modules that support batch program generation."""

    signature_class: Type[dspy.Signature] = None  # Override in subclass

    def __init__(self, sig: Optional[Type[dspy.Signature]] = None, **kwargs):
        # If a dynamic signature is passed, use it, otherwise fall back to class attribute
        self.signature_class = sig or self.signature_class
        if self.signature_class is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define signature_class"
            )
        super().__init__(**kwargs)

    @staticmethod
    def doc_from_markdown(md_path: str):
        """Load markdown file content as instructions for a DSPy signature."""
        def decorator(cls):
            path = Path(md_path)
            if not path.exists():
                raise FileNotFoundError(f"Markdown file not found: {md_path}")

            with path.open("r", encoding="utf-8") as f:
                content = f.read()
                # Avoid over-aggressive space collapsing; just strip trailing spaces
                clean = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)
                cls.instructions = clean
                cls.__doc__ = clean

            return cls
        return decorator

    @staticmethod
    def make_signature_class(
        base_cls: Type[dspy.Signature],
        md_path: str,
        name: Optional[str] = None
    ) -> Type[dspy.Signature]:
        """Create a signature class with instructions from a markdown file."""
        name = name or f"{base_cls.__name__}_{Path(md_path).stem}"

        @ProgramGenerator.doc_from_markdown(md_path)
        class DynamicSignature(base_cls):
            pass

        DynamicSignature.__name__ = name
        DynamicSignature.__qualname__ = name
        return DynamicSignature


    @classmethod
    def save_programs(
        cls,
        prompt_dir: str,
        program_dir: str,
        lm: Optional[dspy.LM] = None,
    ):
        """Generate and save multiple program variants from markdown prompts."""
        if cls.signature_class is None:
            raise NotImplementedError(
                f"{cls.__name__} must define signature_class"
            )

        cls.prompts_to_programs(
            sig_class=cls.signature_class,
            module_factory=cls,
            prompt_dir=prompt_dir,
            program_dir=program_dir,
            lm=lm
        )

    @staticmethod
    def prompts_to_programs(
        sig_class: Type[dspy.Signature],
        module_factory: Callable,
        prompt_dir: str,
        program_dir: str,
        lm: Optional[dspy.LM] = None,
    ) -> None:
        """Process multiple markdown prompts in a directory and save compiled modules."""
        prompt_path = Path(prompt_dir)
        program_path = Path(program_dir)

        if not prompt_path.exists():
            raise NotADirectoryError(f"Prompt directory not found: {prompt_dir}")

        program_path.mkdir(parents=True, exist_ok=True)

        md_files = sorted(prompt_path.glob("*.md"))
        if not md_files:
            print(f"No markdown files found in {prompt_dir}")
            return

        for md_file in md_files:
            ProgramGenerator.prompt_to_program(
                sig_class=sig_class,
                module_factory=module_factory,
                prompt_file=str(md_file),
                program_dir=program_dir,
                lm=lm
            )

    @staticmethod
    def prompt_to_program(
        sig_class: Type[dspy.Signature],
        module_factory: Callable,
        prompt_file: str,
        program_dir: str,
        lm: Optional[dspy.LM] = None
    ) -> None:
        """Process a single markdown prompt file and save a compiled module."""
        prompt_path = Path(prompt_file)
        program_path = Path(program_dir)

        if not prompt_path.exists() or prompt_path.suffix != ".md":
            raise FileNotFoundError(f"Markdown file not found: {prompt_file}")

        program_path.mkdir(parents=True, exist_ok=True)

        try:
            stem = prompt_path.stem
            sig = ProgramGenerator.make_signature_class(
                sig_class, str(prompt_path), name=stem
            )
            mod = module_factory(sig)
            mod.set_lm(lm)

            output_path = program_path / stem
            mod.save(str(output_path), save_program=True)

        except Exception as e:
            print(f"✗ Failed to process {prompt_path.name}: {e}")

    @classmethod
    def load_programs(cls, program_dir: str, lm: Optional[dspy.LM] = None) -> List[dspy.Module]:
        """Load compiled modules from a directory."""
        program_path = Path(program_dir)
        if not program_path.exists():
            raise NotADirectoryError(f"Program directory not found: {program_dir}")
        program_names = cls.list_saved_programs(program_dir)
        print(program_names)
        modules = [dspy.load(program_name) for program_name in program_names]
        if lm is not None:
            for module in modules:
                module.set_lm(lm)
        return modules

    @staticmethod
    def list_saved_programs(program_dir: str) -> List[str]:
        program_path = Path(program_dir)
        if not program_path.exists():
            raise NotADirectoryError(
                f"Program directory not found: {program_dir}"
            )

        names: List[str] = []
        for child in program_path.iterdir():
            if child.name.startswith("."):
                continue

            if child.is_dir():
                if (child / "program.pkl").exists():
                    names.append(program_dir + "/" + child.name)
            elif child.is_file() and child.suffix.lower() in {
                ".json",
                ".yaml",
                ".yml",
                ".pkl",
            }:
                names.append(program_dir + "/" + child.name)

        return sorted(set(names), key=str.lower)


class ProgramLoader(ProgramGenerator):
    def __init__(
        self,
        base_module: dspy.Module,
        sig: Optional[Type[dspy.Signature]] = None,
        lm: Optional[dspy.LM] = None,
    ):
        self.base_module = base_module
        self.lm = lm

        if sig is not None:
            self.signature_class = sig
        elif hasattr(base_module, "signature_class"):
            self.signature_class = base_module.signature_class
        else:
            self.signature_class = self._discover_signature(base_module)

        if self.signature_class is None:
            raise ValueError(f"Could not determine signature_class for {base_module}")

        self._patch_predictor_signature()

        super().__init__(sig=self.signature_class)

    def _patch_predictor_signature(self):
        """Finds and replaces the signature of any dspy.Predict submodule."""
        for name, attr in inspect.getmembers(self.base_module):
            if isinstance(attr, dspy.Predict):
                new_predictor = dspy.Predict(self.signature_class, **attr.config)
                setattr(self.base_module, name, new_predictor)

    def _discover_signature(self, module: dspy.Module) -> Optional[Type[dspy.Signature]]:
        for _, attr in inspect.getmembers(module):
            if isinstance(attr, dspy.Predict):
                return attr.signature
        return None

    def forward(self, *args, **kwargs):
        with dspy.context(lm=self.lm):
            return self.base_module(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.base_module, item)

    def save_instance_programs(
        self,
        prompt_dir: str,
        program_dir: str,
        lm: Optional[dspy.LM] = None,
        clone_base: bool = True,
    ):
        lm_to_use = lm or self.lm

        def factory(sig: Type[dspy.Signature]) -> dspy.Module:
            base = copy.deepcopy(self.base_module) if clone_base else self.base_module
            return ProgramLoader(base, sig=sig, lm=lm_to_use)

        ProgramGenerator.prompts_to_programs(
            sig_class=self.signature_class,
            module_factory=factory,
            prompt_dir=prompt_dir,
            program_dir=program_dir,
            lm=lm_to_use,
        )
