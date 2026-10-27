"""Keep frozen weights intact during destructive publication diagnostics."""


def skip_list_for_action(args, action, skip_list):
    # Snapshots/comparisons still include the frozen tensors: they must remain
    # exactly equal while only the language model is republished.
    if (
        action == "reset_tensors"
        and getattr(args, "custom_generate_function_path", None)
        == "open_instruct.miles.datasets.vision_rollout.generate"
    ):
        return list(dict.fromkeys([*(skip_list or []), "vision_tower.", "connector."]))
    return skip_list
