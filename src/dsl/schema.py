from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Condition(StrictModel):
    path: str
    operator: Literal["eq", "ne", "gt", "ge", "lt", "le", "exists", "in", "contains"]
    value: Any = None


class ValueSource(StrictModel):
    from_: str | None = Field(default=None, alias="from")
    value: Any = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "ValueSource":
        has_from = "from_" in self.model_fields_set
        has_value = "value" in self.model_fields_set
        if has_from == has_value:
            raise ValueError("exactly one of 'from' or 'value' is required")
        return self


class CopyStep(StrictModel):
    op: Literal["copy"]
    from_: str = Field(alias="from")
    to: str


class ConstantStep(StrictModel):
    op: Literal["constant"]
    to: str
    value: Any


class CastStep(StrictModel):
    op: Literal["cast"]
    from_: str = Field(alias="from")
    to: str
    target_type: Literal["string", "integer", "number", "boolean"]


class DefaultStep(StrictModel):
    op: Literal["default"]
    from_: str = Field(alias="from")
    to: str
    value: Any


class CoalesceStep(StrictModel):
    op: Literal["coalesce"]
    sources: list[str] = Field(min_length=1)
    to: str


class MapEnumStep(StrictModel):
    op: Literal["map_enum"]
    from_: str = Field(alias="from")
    to: str
    mapping: dict[str, Any]
    default: Any = None


class ScaleStep(StrictModel):
    op: Literal["scale"]
    from_: str = Field(alias="from")
    to: str
    factor: float
    offset: float = 0.0


class FormatDateStep(StrictModel):
    op: Literal["format_date"]
    from_: str = Field(alias="from")
    to: str
    input_format: str
    output_format: str


class ConcatStep(StrictModel):
    op: Literal["concat"]
    sources: list[str] = Field(min_length=1)
    to: str
    separator: str = ""


class SplitStep(StrictModel):
    op: Literal["split"]
    from_: str = Field(alias="from")
    to: str
    separator: str
    index: int


class NestStep(StrictModel):
    op: Literal["nest"]
    to: str
    fields: dict[str, str] = Field(min_length=1)


class FlattenStep(StrictModel):
    op: Literal["flatten"]
    from_: str = Field(alias="from")
    to: str


class ConditionalStep(StrictModel):
    op: Literal["conditional"]
    condition: Condition
    to: str
    then: ValueSource
    otherwise: ValueSource


ScalarStep = Annotated[
    Union[
        CopyStep,
        ConstantStep,
        CastStep,
        DefaultStep,
        CoalesceStep,
        MapEnumStep,
        ScaleStep,
        FormatDateStep,
        ConcatStep,
        SplitStep,
        NestStep,
        FlattenStep,
        ConditionalStep,
    ],
    Field(discriminator="op"),
]


class MapArrayStep(StrictModel):
    op: Literal["map_array"]
    from_: str = Field(alias="from")
    to: str
    steps: list[ScalarStep] = Field(min_length=1)


class FilterArrayStep(StrictModel):
    op: Literal["filter_array"]
    from_: str = Field(alias="from")
    to: str
    condition: Condition


Step = Annotated[Union[ScalarStep, MapArrayStep, FilterArrayStep], Field(discriminator="op")]


class Program(StrictModel):
    steps: list[Step] = Field(min_length=1)
