"""CRAG exception hierarchy."""


class CRAGError(Exception):
    """Base error for all CRAG operations."""


class ConfigError(CRAGError):
    """Invalid or missing configuration."""


class ParserError(CRAGError):
    """Document parsing failed."""


class UnsupportedFormatError(ParserError):
    """No registered parser handles the given file."""

    def __init__(self, path: str, extension: str | None = None) -> None:
        self.path = path
        self.extension = extension
        suffix = f" (extension: {extension})" if extension else ""
        super().__init__(f"unsupported file format: {path}{suffix}")


class ParseFailedError(ParserError):
    """Parser matched the file but failed during extraction."""

    def __init__(self, path: str, reason: str, *, cause: Exception | None = None) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"failed to parse {path}: {reason}")
        self.__cause__ = cause
