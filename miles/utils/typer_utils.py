"""Expose dataclass fields as Typer options for repository scripts."""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Annotated, Any, TypeVar, cast, get_args, get_origin, get_type_hints, overload

import typer

_F = TypeVar("_F", bound=Callable[..., object])
SCRIPT_ENV_VAR_PREFIX = "MILES_SCRIPT_"


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
            config = _construct(config_type, values)
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
    if get_origin(annotation) is tuple and get_args(annotation)[1:] == (Ellipsis,):
        annotation = list[get_args(annotation)[0]]
        if isinstance(default, tuple):
            default = list(default)
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


def _construct(config_type: type, values: Mapping[str, object]) -> Any:
    converted = dict(values)
    for name, annotation in get_type_hints(config_type).items():
        if name in converted and get_origin(annotation) is tuple and isinstance(converted[name], list):
            converted[name] = tuple(converted[name])
    return config_type(**converted)


def dataclass_from_env(
    dataclass_cls: type, *, env_var_prefix: str = SCRIPT_ENV_VAR_PREFIX, overrides: Mapping[str, object] | None = None
) -> Any:
    """Use the same Click option conversion for environment-only launch configuration."""
    annotations = get_type_hints(dataclass_cls)
    overrides = dict(overrides or {})
    result = []

    def build(**values):
        result.append(_construct(dataclass_cls, {**values, **overrides}))

    options = [
        _option(field, annotation=annotations[field.name], env_var_prefix=env_var_prefix)
        for field in dataclasses.fields(dataclass_cls)
        if field.init and field.name not in overrides
    ]
    build.__signature__ = inspect.Signature(options)
    build.__annotations__ = {option.name: option.annotation for option in options}
    app = typer.Typer(add_completion=False)
    app.command()(build)
    typer.main.get_command(app)(args=[], standalone_mode=False)
    return result[0]


def _print_arguments(config: Any) -> None:
    rows = []
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        rows.append((field.name, str(value.value if isinstance(value, Enum) else value)))
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
