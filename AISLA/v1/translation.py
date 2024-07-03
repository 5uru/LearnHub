import re

import torch
from sentence_splitter import SentenceSplitter
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

model_load_name = "jonathansuru/dioula_saved_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name, low_cpu_mem_usage=True)
if torch.cuda.is_available():
    model.cuda()
tokenizer = NllbTokenizer.from_pretrained(model_load_name)


def sentenize_with_fillers(text, splitter, fix_double_space=True, ignore_errors=False):
    """Apply a sentence splitter and return the sentences and all separators before and after them"""
    if fix_double_space:
        text = re.sub(" +", " ", text)
    sentences = splitter.split(text)
    fillers = []
    i = 0
    for sentence in sentences:
        start_idx = text.find(sentence, i)
        if ignore_errors and start_idx == -1:
            # print(f"sent not found after {i}: `{sentence}`")
            start_idx = i + 1
        assert start_idx != -1, f"sent not found after {i}: `{sentence}`"
        fillers.append(text[i:start_idx])
        i = start_idx + len(sentence)
    fillers.append(text[i:])
    return sentences, fillers


def translate(
    text,
    src_lang,
    tgt_lang,
    max_length="auto",
    num_beams=4,
    by_sentence=True,
    **kwargs,
):
    """Translate a text sentence by sentence, preserving the fillers around the sentences."""
    if by_sentence:
        sents, fillers = sentenize_with_fillers(
            text, splitter=SentenceSplitter("fr"), ignore_errors=True
        )
    else:
        sents = [text]
        fillers = ["", ""]

    results = []
    for sent, sep in zip(sents, fillers):
        results.extend(
            (
                sep,
                translate_single(
                    sent,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    max_length=max_length,
                    num_beams=num_beams,
                    **kwargs,
                ),
            )
        )
    results.append(fillers[-1])
    return "".join(results)


def translate_single(
    text,
    src_lang,
    tgt_lang,
    max_length="auto",
    num_beams=4,
    n_out=None,
    **kwargs,
):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if max_length == "auto":
        max_length = int(32 + 2.0 * encoded.input_ids.shape[1])
    generated_tokens = model.generate(
        **encoded.to(model.device),
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=n_out or 1,
        **kwargs,
    )
    out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return out[0] if isinstance(text, str) and n_out is None else out


def main(
    text,
    src_lang,
    tgt_lang,
    max_length="auto",
    num_beams=4,
    by_sentence=True,
    **kwargs,
):
    if by_sentence:
        return translate(
            text, src_lang, tgt_lang, max_length, num_beams, by_sentence, **kwargs
        )
    else:
        return translate_single(
            text, src_lang, tgt_lang, max_length, num_beams, **kwargs
        )
