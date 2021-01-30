import torch
import string

def prediction_from_model(model, tokenizer, tokens, n=100):
    """collect replace words for [MASK] from model predictions

    Args:
        model:  support huggingface models
        tokenizer: support huggingface tokenizer
        tokens: list of str
        n: return n candidates for each mask word.

    Returns:
        list, shape (mask_num * n)
    """
    original_text = ' '.join(x for x in tokens)

    tokenized_text = tokenizer.tokenize(original_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    # convert to tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    replace_words_for_masks = []
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
        prediction = predictions[0][tokens_tensor == tokenizer.mask_token_id]  # mask_num * vocab_size
        indices = prediction.argsort(dim=-1, descending=True)
        for i in range(indices.size(0)):
            replace_words_for_mask = []
            for j in range(indices.size(1)):
                if len(replace_words_for_mask) < n:
                    replace_word = tokenizer.convert_ids_to_tokens(indices[i][j].item())
                    if replace_word in string.punctuation or replace_word.startswith('##') \
                            or len(replace_word) == 1 or replace_word.startswith('.') or replace_word.startswith('['):
                        continue
                    replace_words_for_mask.append(replace_word)

            replace_words_for_masks.append(replace_words_for_mask)
    return replace_words_for_masks