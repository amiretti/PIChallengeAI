class RagError(Exception):
    """Base class for errors raised by this application."""


class DocumentNotFound(RagError):
    pass


class DocumentReadError(RagError):
    pass
