from pathlib import Path
from typing import List, Union


def get_video_paths(path_to_video_folder: Union[str, Path]) -> List[Path]:
    """Search the folder for 'mp4' files (case insensitive) and return them as a list"""

    list_of_video_paths = list(Path(path_to_video_folder).glob("*.mp4")) + list(
        Path(path_to_video_folder).glob("*.MP4")
    )
    unique_list_of_video_paths = get_unique_list(list_of_video_paths)

    return sorted(unique_list_of_video_paths, key=lambda p: str(p).lower())


def get_unique_list(list: list) -> list:
    """Return a list of the unique elements from input list"""
    unique_list = []
    [unique_list.append(element) for element in list if element not in unique_list]

    return unique_list
