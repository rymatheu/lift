"""Transformer implementations and the tokenizer they share."""

MODEL_CHOICES = ("gemma4", "gemma4-12b", "smollm2")


def load(name: str):
    """Return the model module for a --model choice.

    Each module exposes build_model(), get_contextual_embeddings() and
    save_initial_embeddings() with matching signatures.
    """
    if name == "gemma4":
        from lift.models import gemma4 as mod
    elif name == "gemma4-12b":
        from lift.models import gemma4_12b as mod
    elif name == "smollm2":
        from lift.models import smollm2 as mod
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_CHOICES}")
    return mod
