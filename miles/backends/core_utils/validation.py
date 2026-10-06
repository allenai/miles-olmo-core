"""Model interchange validation independent of the application CLI."""

import difflib
import math

InputError = ValueError


def integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        requirement = "positive integer" if minimum == 1 else "nonnegative integer"
        raise InputError(f"{name} must be a {requirement}; got {value!r}.")
    return value


def number(value, name, *, minimum=0, maximum=None, exclusive_min=False, exclusive_max=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    bounds = f"{'>' if exclusive_min else '>='} {minimum}"
    if maximum is not None:
        bounds += f" and {'<' if exclusive_max else '<='} {maximum}"
    if (
        not finite
        or value < minimum
        or (exclusive_min and value == minimum)
        or (maximum is not None and (value > maximum or (exclusive_max and value == maximum)))
    ):
        raise InputError(f"{name} must be a finite number {bounds}; got {value!r}.")
    return value


def mapping(value, name):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InputError(f"{name} must be a table/object with named fields.")
    return value


def fields(value, name, allowed):
    mapping(value, name)
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        hints = []
        for key in unknown:
            matches = difflib.get_close_matches(key, sorted(allowed), n=1)
            if matches:
                hints.append(f"{key!r}: did you mean {matches[0]!r}?")
        raise InputError(f"Unknown {name} fields: {unknown}. " + (" ".join(hints) or "Remove these fields."))
