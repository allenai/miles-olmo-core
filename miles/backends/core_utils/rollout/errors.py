"""Generation lifecycle errors shared by Core rollout workers."""

# Application providers supported by the finite-drain and training-feedback protocol.
FINITE_POOL_PROVIDERS = ("pools",)
# Drop handlers notify feedback providers before retiring recovery ownership.
EXPLICIT_DISCARD_FEEDBACK = True


class GenerationInterrupted(RuntimeError):
    """A producer must quiesce before resuming under a new policy."""


class GenerationRequestTimeout(TimeoutError):
    """One generation exceeded its deadline; its entire prompt group can retry."""


class PromptPoolExhausted(RuntimeError):
    """All finite prompt pools are drained; no complete training batch remains."""
