---
tags: [config, fortran, icon]
---

# [Configuration Classes]

- **Status**: valid
- **Authors**: Rico Häuselmann (@DropD)
- **Created**: 2026-06-17
- **Updated**: 2026-06-17

(Wh)Y-statement: In the context of [use case/user story u], facing [concern c] we decided for [option o] and neglected [other options], to achieve [system qualities/desired consequences], accepting [downside/undesired consequences].

While making Icon4Py user-configurable, facing code duplication between modules, we decided to provide a standard way to declaratively encode common attributes of configuration options alongside utilities that can then be reused across modules.

## Context

We notice that currently, each module has a class, which contains it's configuration. Each of these classes knows how to construct itself from a dictionary that represents an ICON namelist. The mapping is contained implicitly in a classmethod. Usually the class attributes that have an equivalent in ICON are annotated with documentation comments (where to find in ICON namelists and what it does).

Really, given the mapping for each configuration option in each of those classes, the logic is the same for all of them.

Extrapolating from this case of duplication we find the same for:

- describing configuration options to users (currently only documentation comments exist for this)
- validation (currently partially done in monolithic, imperative validation methods)

## Decision

The chosen pattern is for configuration classes to look as Follows

```python
import dataclasses
import typing

from icon4py.model.common.config import options as common_conf_opt


@dataclasses.dataclass
class MyModuleConfig:
    some_configuration_option: typing.Annotated[
        int,
        common_conf_opt.ConfigOption(
            description="This is help telling the user what `some_configuration_option` does.",
            icon_equivalent=common_conf_opt.IconOption(
                name="isomcfgop",
                path=("diffusion_nml",),  # for nested sections: ("toplevel_nml", "sublevel_nml")
                read_from_icon=True,  # optional (default: True), whether to actually read this from ICON namelists
                list_to_value=False,  # optional (default: False), whether to map a list of values in ICON namelists to a single value
            ),
        ),
    ] = 4  # default value
```

The pattern is meant to be extended in the future as the need arises. One example would be validation:

```python
def validate_some_configuration_option_with_some_flag(
    some_configuration_option: int, other_option: bool
) -> None:
    if not other_option:
        raise ConfigValidationError(
            f"'some_configuration_option' should only be set if 'other_option' is True."
        )


...


@dataclasses.dataclass
class MyModuleConfig:
    some_configuration_option: typing.Annotated[
        int,
        ConfigOption(
            ...,
            validators=(
                DependsOn("some_flag", condition=validate_some_configuration_option_with_some_flag),
                Min(0),
                Max(59000),
                ...,
            ),
        ),
    ] = 4
```

## Consequences

Code for mapping from ICON inputs to Icon4Py configurations can be shared between modules.

It becomes possible to automatically provide helpful (for users) information for all available configuration options (for a given set of modules).

Different user facing utilities for configuring / inspecting Icon4Py runs will not have to duplicate logic for

- mapping to / from ICON (fortran)
- validation
- documentation
- etc

The current solution will not achieve maximum separation isolation between Icon4Py internals and their mapping to and from ICON concepts. However it provides a structured path towards it.

Another downside is a certain loss of type checking when using the provided utilities. This can be made up for in tests.

## Alternatives Considered

### Separating (fortran-) ICON mapping information from the rest

- Pro: allow free evolution of Icon4Py internals, decoupled from ICON
- Con: more complicated, less (for now) helpful information immediately available

This could be achieved, however, by simply moving all the instances of `IconOption` out of `ConfigOption` annotations and into a separate (per-module) dictionary, keyed by the name (or class AND name) of the option it belongs to.

### Use `dataclasses.field` instead of `typing.Annotated`

```python
class MyModuleConfig:
    some_configuration_option: int = dataclasses.field(
        metadata={"config_option": ConfigOption(...)}, default=4
    )
```

A wrapper around `dataclasses.field` could improve the readability of this option.

This option is functionally equivalent. The one difference is that `typing.Annotated` is lazily evaluated, while `dataclasses.field` is not. It was not chosen because syntactically it groups the annotations with the default value as opposed to the type.

## References

- [Python Documentation for `typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated)
