from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_RIGHT_IRIS,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_FACE_OVAL,
)


def get_unique_indices_and_names(
    connections: frozenset[tuple[int, int]],
    prepend_string: str,
) -> tuple[frozenset[int], list[str]]:
    indices: set[int] = set()
    for start, end in connections:
        indices.add(start)
        indices.add(end)
    sorted_indices = sorted(indices)
    names = [f"{prepend_string}_{i}" for i in sorted_indices]
    return frozenset(indices), names


MEDIAPIPE_FACE_LIPS_INDICIES, MEDIAPIPE_LIPS_NAMES = get_unique_indices_and_names(FACEMESH_LIPS, "face.lips")
MEDIAPIPE_FACE_LEFT_EYE_INDICIES, MEDIAPIPE_LEFT_EYE_NAMES = get_unique_indices_and_names(FACEMESH_LEFT_EYE, "face.left_eye")
MEDIAPIPE_FACE_RIGHT_EYE_INDICIES, MEDIAPIPE_RIGHT_EYE_NAMES = get_unique_indices_and_names(FACEMESH_RIGHT_EYE, "face.right_eye")
MEDIAPIPE_FACE_LEFT_EYEBROW_INDICIES, MEDIAPIPE_LEFT_EYEBROW_NAMES = get_unique_indices_and_names(FACEMESH_LEFT_EYEBROW, "face.left_eyebrow")
MEDIAPIPE_FACE_RIGHT_EYEBROW_INDICIES, MEDIAPIPE_RIGHT_EYEBROW_NAMES = get_unique_indices_and_names(FACEMESH_RIGHT_EYEBROW, "face.right_eyebrow")
MEDIAPIPE_FACE_FACE_OVAL_INDICIES, MEDIAPIPE_FACE_OVAL_NAMES = get_unique_indices_and_names(FACEMESH_FACE_OVAL, "face.face_oval")
MEDIAPIPE_FACE_RIGHT_IRIS_INDICIES, MEDIAPIPE_RIGHT_IRIS_NAMES = get_unique_indices_and_names(FACEMESH_RIGHT_IRIS, "face.right_iris")
MEDIAPIPE_FACE_LEFT_IRIS_INDICIES, MEDIAPIPE_LEFT_IRIS_NAMES = get_unique_indices_and_names(FACEMESH_LEFT_IRIS, "face.left_iris")

MEDIAPIPE_FACE_CONTOURS_INDICIES = frozenset().union(
    MEDIAPIPE_FACE_LIPS_INDICIES,
    MEDIAPIPE_FACE_LEFT_EYE_INDICIES,
    MEDIAPIPE_FACE_RIGHT_EYE_INDICIES,
    MEDIAPIPE_FACE_LEFT_EYEBROW_INDICIES,
    MEDIAPIPE_FACE_RIGHT_EYEBROW_INDICIES,
    MEDIAPIPE_FACE_FACE_OVAL_INDICIES,
    MEDIAPIPE_FACE_RIGHT_IRIS_INDICIES,
    MEDIAPIPE_FACE_LEFT_IRIS_INDICIES,
)
MEDIAPIPE_FACE_CONTOURS_NAMES = (
    MEDIAPIPE_LIPS_NAMES
    + MEDIAPIPE_LEFT_EYE_NAMES
    + MEDIAPIPE_RIGHT_EYE_NAMES
    + MEDIAPIPE_LEFT_EYEBROW_NAMES
    + MEDIAPIPE_RIGHT_EYEBROW_NAMES
    + MEDIAPIPE_FACE_OVAL_NAMES
    + MEDIAPIPE_RIGHT_IRIS_NAMES
    + MEDIAPIPE_LEFT_IRIS_NAMES
)

if len(MEDIAPIPE_FACE_CONTOURS_INDICIES) != len(MEDIAPIPE_FACE_CONTOURS_NAMES):
    raise ValueError(
        f"Expected indices and names to have equal length, "
        f"got {len(MEDIAPIPE_FACE_CONTOURS_INDICIES)} indices and {len(MEDIAPIPE_FACE_CONTOURS_NAMES)} names"
    )
