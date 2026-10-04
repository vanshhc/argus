"""Register this project's ports with Apple's exporter at run time."""

from coreai_models.models import registry

from coreai_ports.llama_ios import LlamaForCausalLMForiOS

PORTS = {"llama": LlamaForCausalLMForiOS}


def register() -> None:
    """Add iOS ports to Apple's model registry. Never replace one of Apple's entries."""
    entries = registry._get_registry()
    for model_type, ios_class in PORTS.items():
        existing = entries.get(model_type)
        if existing is None:
            entries[model_type] = registry.ModelEntry(ios_class=ios_class)
        elif existing.ios_class is None:
            existing.ios_class = ios_class
        elif existing.ios_class is not ios_class:
            raise RuntimeError(f"Apple's exporter already has an iOS class for {model_type!r}")
