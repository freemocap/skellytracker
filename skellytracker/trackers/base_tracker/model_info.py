from typing import Dict, List, Optional


class ModelInfo(dict):
    model_name: str # TODO: rename to tracker_name to avoid pydantic 2 conflict?
    landmark_names: List[str]
    num_tracked_points: int
    tracked_object_names: Optional[list] = None
    virtual_markers_definitions: Optional[Dict[str, Dict[str, List[str | float]]]] = None
    segment_connections: Optional[Dict[str, Dict[str, str]]] = None
    center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None
    joint_hierarchy: Optional[Dict[str, List[str]]] = None
