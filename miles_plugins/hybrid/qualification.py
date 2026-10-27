"""Align SGLang's prompt-logprob suffix and decode logprobs by token position."""


def align_logprobs(prompt, generated, metadata):
    # The pinned SGLang prepends None to the returned suffix even when
    # logprob_start_len is nonzero; dropping it must not shift token positions.
    inputs = metadata["input_token_logprobs"]
    outputs = metadata["output_token_logprobs"]
    offset = len(prompt) - len(inputs)
    if offset < 0 or [entry[1] for entry in inputs] != prompt[offset:]:
        raise AssertionError(f"Prompt logprob token IDs are not a prompt suffix: offset={offset}, entries={inputs[:3]}")
    if [entry[1] for entry in outputs] != generated or any(entry[0] is None for entry in outputs):
        raise AssertionError("Decode logprobs must cover every generated token in order")
    positioned = [(offset + index, entry[0]) for index, entry in enumerate(inputs) if entry[0] is not None]
    positioned.extend((len(prompt) + index, entry[0]) for index, entry in enumerate(outputs))
    if not positioned or any(position < 1 for position, _ in positioned):
        raise AssertionError("Log probabilities require a preceding token")
    return [position for position, _ in positioned], [value for _, value in positioned]
