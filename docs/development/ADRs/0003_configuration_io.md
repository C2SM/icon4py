---
tags: [config, io, yaml]
---

# [Configuration IO]

- **Status**: valid
- **Authors**: Rico Häuselmann (@DropD)
- **Created**: 2026-07-29
- **Updated**: 2026-07-29

While implementing and discussing the first iteration of ICON4Py's own configuration files, the following design choices were made.

## Context

The configuration file is the primary way for a normal user to interact with ICON4Py, hence this iteration has been preceded by experiments and discussions. This ADR is a record of the results.

The following features / trade-offs were discussed and investigated

- compatibility with ICON
- configuration file format choice
- usability of the configuration file format
- potential lock-in into specific frameworks
- round-tripping may or may not change a configuration file

## Decision

### ICON compatibility

ICON4Py is currently able to read ICON configurations (namelists) on a best-effort basis (not all options supported). This may be replaced by a tool to convert namelists to ICON4Py configuration files

ICON4Py configuration files need not be readable by ICON or any other application.

### Configuration file format

Desired: human readable well-defined format.

Current choice: YAML as default, other options supported by `cattrs`, the library we currently use.

Change if:

 - a demonstrably better format is identified

### Usability - Duplicated Options

At the time of writing this ADR, some component configuration classes contain attributes which should be configured only once for the whole experiment. They also provide default values for those options. This makes the configuration format obviously error prone.

We decided to remove these duplicated options from the components, as they were historically necessary but serve only convenience now. This is a work in progress.

### Usability - Round Trips

Ideally, round-tripping configuration files should not change them.

Current state:

- true for configuration files written by ICON4Py, not true in general
- the `pyyaml` library, which powers our YAML support does not support the general case

Change if:

- if this is considered worth the (small) effort, `pyyaml` could be replaced with `ruamel` or another library with round-tripping support

### Avoiding lock-in

If possible avoid frameworks that only work with data structures they themselves provide.

If possible avoid frameworks that require custom serialization / de-serialization logic to be defined inside the data structures to be customized.

We decide to start out using `cattrs` because it

- can work with the existing configuration data classes
- serialization and de-serialization can be customized without touching the classes we customize for
- was already a transitive dependency at the time

This decision is to be revisited if we find ourselves writing a lot of code to enable features we could get more easily by switching library / framework.

## Consequences

- we keep flexibility
- we reduce technical debt rather than increasing configuration system complexity
- we keep user experience "good enough for now"
- component development does not require additional framework knowledge

## Alternatives Considered

### Using `omegaconf`

An earlier iteration ([#936](https://github.com/C2SM/icon4py/pull/936)) used `omegaconf`.

- required little change in existing configuration classes
- avoided cleaning up the duplicated configuration options in component configuration classes by automatically synchronizing the component's values with the top-level (if given)
- introduced additional indirection: a `ConfigManager` class was built from the configuration file, which then in turn built the configuration classes.
- introduced the need to wrap configuration classes in `ConfigHandler`s in some cases

All in all it traded complexity for the feature that config classes can define options that refer to options in a config class higher up the hierarchy.
