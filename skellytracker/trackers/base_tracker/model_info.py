from typing import Optional


class ModelInfo:
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


if __name__ == "__main__":
    class TestModelInfo(ModelInfo):
        landmark_names = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
        ]
        connections = [
            ["a", "b"],
            ["b", "c"],
            ["c", "d"],
            ["d", "e"],
            ["e", "f"],
            ["f", "g"],
            ["g", "h"],
            ["h", "i"],
            ["i", "j"],
            ["j", "k"],
            ["k", "l"],
            ["l", "m"],
            ["m", "n"],
        ]
        num_tracked_points = 14
        tracked_object_names = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
        ]
        segment_names = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
        ]
        joint_connections = [
            ["a", "b"],
            ["b", "c"],
            ["c", "d"],
            ["d", "e"],
            ["e", "f"],
            ["f", "g"],
            ["g", "h"],
            ["h", "i"],
            ["i", "j"],
            ["j", "k"],
            ["k", "l"],
            ["l", "m"],
            ["m", "n"],
        ]
        segment_COM_lengths = [
            0.5,
            0.5,
            0.436,
            0.436,
            0.430,
            0.430,
            0.506,
            0.506,
            0.433,
            0.433,
            0.433,
            0.433,
            0.5,
            0.5,
        ]
        segment_COM_percentages = [
            0.081,
            0.497,
            0.028,
            0.028,
            0.016,
            0.016,
            0.006,
            0.006,
            0.1,
            0.1,
            0.0465,
            0.0465,
            0.0145,
            0.0145,
        ]
        names_and_connections_dict = {
            "tracked_points": {
                "names": landmark_names,
                "connections": connections,
            },
        }
        virtual_marker_definitions_dict = {
            "o": {
                "marker_names": ["a", "b"],
                "marker_weights": [0.5, 0.5],
            },
            "p": {
                "marker_names": ["b", "c"],
                "marker_weights": [0.5, 0.5],
            },
            "q": {
                "marker_names": ["d", "e"],
                "marker_weights": [0.5, 0.5],
            },
        }
        skeleton_schema = {
            "tracked_points": {
                "names": landmark_names,
                "connections": connections,
            },
        }
        joint_hierarchy = {
            "tracked_points": {
                "names": landmark_names,
                "connections": connections,
            },
        }
