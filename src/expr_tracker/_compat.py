from typing import Any, Type

from pydantic import BaseModel
from pydantic.version import VERSION as PYDANTIC_VERSION
from typing_extensions import Literal

PYDANTIC_VERSION_MINOR_TUPLE = tuple(int(x) for x in PYDANTIC_VERSION.split(".")[:2])
PYDANTIC_V2 = PYDANTIC_VERSION_MINOR_TUPLE[0] == 2

Url: Type[Any]

if PYDANTIC_V2:
    from pydantic_core import PydanticUndefinedType
    from pydantic_core import Url as Url

    UndefinedType = PydanticUndefinedType

    def _model_dump(
        model: BaseModel, mode: Literal["json", "python"] = "json", **kwargs: Any
    ) -> Any:
        return model.model_dump(mode=mode, **kwargs)
else:
    from pydantic import AnyUrl as Url  # noqa: F401
    from pydantic.fields import (  # type: ignore[no-redef, attr-defined]
        UndefinedType as UndefinedType,  # noqa: F401
    )

    def _model_dump(
        model: BaseModel, mode: Literal["json", "python"] = "json", **kwargs: Any
    ) -> Any:
        return model.dict(**kwargs)
