"""Lazy backend exports; shared configuration helpers do not load the HF trainer."""


def __getattr__(name):
    if name == "FSDPTrainRayActor":
        from .actor import FSDPTrainRayActor

        return FSDPTrainRayActor
    if name == "load_fsdp_args":
        from .arguments import load_fsdp_args

        return load_fsdp_args
    raise AttributeError(name)
