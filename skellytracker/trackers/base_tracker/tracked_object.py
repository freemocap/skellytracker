from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrackedObject:
    """
    A dataclass for storing information about a tracked object in a single image/frame
    """

    object_id: str
    pixel_x: Optional[float] = None
    pixel_y: Optional[float] = None
    depth_z: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
