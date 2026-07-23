from __future__ import annotations


class ReasonKitError(Exception):
    # Base class for all ReasonKit errors.
    pass


class ConfigurationError(ReasonKitError):
    # Raised at enhance() call time for invalid config -- fails fast.
    pass


class ModelCallError(ReasonKitError):
    # The wrapped fn raised and all retries were exhausted. Critical stages
    # (classify, raw answer) re-raise; non-critical ones degrade (see core._call).

    def __init__(self, message: str, stage: str = "", cause: BaseException | None = None):
        super().__init__(message)
        self.stage = stage
        self.__cause__ = cause
