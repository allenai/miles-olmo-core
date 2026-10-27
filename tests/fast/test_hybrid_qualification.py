import pytest

from miles_plugins.hybrid.qualification import align_logprobs


def test_prompt_suffix_preserves_missing_first_logprob_position():
    positions, values = align_logprobs(
        [11, 12, 13, 14],
        [21, 22],
        {
            "input_token_logprobs": [(None, 13, None), (-2.0, 14, None)],
            "output_token_logprobs": [(-1.0, 21, None), (-3.0, 22, None)],
        },
    )
    assert positions == [3, 4, 5]
    assert values == [-2.0, -1.0, -3.0]


def test_single_token_prompt_has_only_decode_logprobs():
    assert align_logprobs([11], [21], {"input_token_logprobs": [], "output_token_logprobs": [(-1.0, 21, None)]}) == (
        [1],
        [-1.0],
    )


def test_wrong_prompt_tokens_fail_instead_of_silently_shifting():
    with pytest.raises(AssertionError, match="prompt suffix"):
        align_logprobs([11, 12], [21], {"input_token_logprobs": [(-1.0, 99, None)], "output_token_logprobs": [(-1.0, 21, None)]})
