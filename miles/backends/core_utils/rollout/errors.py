"""Generation lifecycle errors shared by Core rollout workers."""


class GenerationInterrupted(RuntimeError):
    """A producer must quiesce before resuming under a new policy."""


class GenerationRequestTimeout(TimeoutError):
    """One generation exceeded its deadline; its entire prompt group can retry."""


class PromptPoolExhausted(RuntimeError):
    """All finite prompt pools are drained; no complete training batch remains."""
