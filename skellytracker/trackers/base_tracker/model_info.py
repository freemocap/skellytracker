from typing import Optional


class ModelInfo(dict):
    landmark_names: Optional[list] = None
    connections: Optional[list] = None
    num_tracked_points: Optional[int] = None
    tracked_object_names: Optional[list] = None
    segment_names: Optional[list] = None
    joint_connections: Optional[list] = None
    segment_COM_lengths: Optional[list] = None
    segment_COM_percentages: Optional[list] = None
    names_and_connections_dict: Optional[dict] = None
    virtual_marker_definitions_dict: Optional[dict] = None
    skeleton_schema: Optional[dict] = None
    joint_hierarchy: Optional[dict] = None
