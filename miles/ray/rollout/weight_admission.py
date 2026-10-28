"""Keep publication diagnostics inside one admission window across local or RPC handles."""

from contextlib import asynccontextmanager


@asynccontextmanager
async def defer_weight_admission(controller):
    """Admit updated worker identities only after the publication body succeeds.

    Each transfer retains its own lock/health protocol. The controller captures
    the identities actually updated and checks them again under its lifecycle
    lock before confirming registration. Keep this context manager on the caller;
    only the begin/finish operations cross the controller RPC boundary.
    """
    await controller.begin_weight_admission()
    try:
        yield
    except BaseException:
        await controller.finish_weight_admission(admit=False)
        raise
    else:
        await controller.finish_weight_admission(admit=True)
