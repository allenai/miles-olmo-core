"""Keep publication diagnostics inside one admission window across local or RPC handles."""

from contextlib import asynccontextmanager

from miles.utils import logger_utils

logger = logger_utils.setup_logger(__name__)


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
        try:
            await controller.finish_weight_admission(admit=False)
        except Exception:
            logger.exception("Admission abort failed; preserving the original publication failure")
        raise
    else:
        await controller.finish_weight_admission(admit=True)
