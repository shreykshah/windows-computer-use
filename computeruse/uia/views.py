from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from computeruse.dom.views import DOMState


class WindowInfo(BaseModel):
    """Represents information about a window"""
    id: int
    title: str
    process_name: str


@dataclass
class WindowState(DOMState):
    """
    Represents the current state of a window
    """
    title: str
    process_name: str
    screenshot: Optional[str] = None
    windows: list[Dict[str, Any]] = field(default_factory=list)


class WindowError(Exception):
    """Base class for all window errors"""
    pass