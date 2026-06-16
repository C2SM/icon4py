import dataclasses


@dataclasses.dataclass
class IconOption:
    name: str
    path: tuple[str, ...]
    list_to_value: bool = False

@dataclasses.dataclass
class ConfigOption:
    description: str
    icon_equivalent: IconOption


def opts_from_config_dataclass(cls: dataclasses.Dataclass) -> dict[str, ConfigOption]:
    #  TODO(ricoh): implement


def construct_config_from_icon_options(cls: T, icon_options: dict[str, Any]) -> T:
    # TODO(ricoh): implement

