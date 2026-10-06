"""Expose dataclass fields as Typer options for repository scripts."""

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import Annotated, Any, TypeVar, cast, get_type_hints, overload

import typer

_F = TypeVar("_F", bound=Callable[..., object])


@overload
def dataclass_cli(func: _F) -> _F: ...


@overload
def dataclass_cli(func: None = None, *, env_var_prefix: str = "MILES_SCRIPT_") -> Callable[[_F], _F]: ...


def dataclass_cli(func: _F | None = None, *, env_var_prefix: str = "MILES_SCRIPT_") -> _F | Callable[[_F], _F]:
    """Call a command with a dataclass populated from CLI options.

    Use ``@dataclass_cli`` for ``MILES_SCRIPT_<FIELD>`` environment defaults,
    or ``@dataclass_cli(env_var_prefix="")`` to disable environment binding.
    Field metadata may supply ``help`` text. Default factories are evaluated
    once when registering the command, matching Typer's static option defaults.
    """

    def register(command: _F) -> _F:
        argument_name = next(iter(inspect.signature(command).parameters))
        config_type = get_type_hints(command)[argument_name]
        assert dataclasses.is_dataclass(config_type)
        annotations = get_type_hints(config_type)
        options = tuple(
            _option(field, annotation=annotations[field.name], env_var_prefix=env_var_prefix)
            for field in dataclasses.fields(config_type)
            if field.init
        )

        @functools.wraps(command)
        def invoke(**values: object) -> object:
            config = config_type(**values)
            _print_arguments(config)
            return command(config)

        invoke.__annotations__ = {option.name: option.annotation for option in options}
        invoke.__signature__ = inspect.Signature(options)  # type: ignore[attr-defined]
        return cast(_F, invoke)

    return register if func is None else register(func)


def _option(field: dataclasses.Field[Any], *, annotation: Any, env_var_prefix: str) -> inspect.Parameter:
    default = field.default
    if field.default_factory is not dataclasses.MISSING:
        default = field.default_factory()
    elif default is dataclasses.MISSING:
        default = inspect.Parameter.empty
    option = typer.Option(
        help=field.metadata.get("help"),
        envvar=f"{env_var_prefix}{field.name.upper()}" if env_var_prefix else None,
    )
    return inspect.Parameter(
        field.name,
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=default,
        annotation=Annotated[annotation, option],
    )


def _print_arguments(config: Any) -> None:
    rows = [(field.name, str(getattr(config, field.name))) for field in dataclasses.fields(config)]
    name_width = max((len(name) for name, _ in rows), default=len("Argument"))
    name_width = max(name_width, len("Argument"))
    border = f"+{'-' * (name_width + 2)}+{'-' * 52}+"
    print(border)
    print(f"| {'Argument':<{name_width}} | {'Value':<50} |")
    print(border)
    for name, value in rows:
        display = value if len(value) <= 50 else f"{value:.47}..."
        print(f"| {name:<{name_width}} | {display:<50} |")
    print(border)
