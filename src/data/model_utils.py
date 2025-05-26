def resize_model_if_needed(tokenizer, model):
    """
    Resizes the model's embedding layer if the tokenizer's vocabulary size
    is larger than the current embedding layer. Useful when using chat template.
    """
    # Get tokenizer and model vocabulary sizes
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.size(0)

    # Check if resizing is needed
    if tokenizer_vocab_size > model_vocab_size:
        print(
            f"Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}."
        )
        model.resize_token_embeddings(tokenizer_vocab_size)
        model.tie_weights()

    return model
