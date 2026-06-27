import os
from datetime import datetime
from pathlib import Path
from typing import List


class Quine:
    """
    A class to represent a Quine.

    A Quine is a computer program which takes no input and produces a copy of its own source code as its only output.
    In this case, the term "Quine" is used a bit more loosely to represent a program that prints out the source code
    in a directory. This can be useful for various purposes, such as feeding the code through a Language-to-Language
    Migration (LLM) tool.

    ...

    Attributes
    ----------
    base_directory : str
        the base directory to start from
    excluded_directories : set
        a set of directories to be excluded
    included_extensions : set
        a set of file extensions to be excluded

    Methods
    -------
    write_to_file(root_directory:str, file_name:str, output_file:object):
        Writes the content of the file to the output markdown file.
    generate_quine():
        Walks through the base directory and its subdirectories, excluding certain directories and file types,
        and writes the Python source code to a Markdown file with headings for each folder and subheadings for each file.
        The source code is enclosed in ```python ``` code blocks.
    """

    def __init__(
        self,
        base_directory: str,
        excluded_directories: List[str],
        included_extensions: List[str],
    ):
        self.base_directory = base_directory
        self.excluded_directories = excluded_directories
        self.included_extensions = included_extensions

    def write_to_file(
        self, root_directory: str, file_name: str, output_file: object
    ) -> None:
        """
        Writes the content of the file to the output markdown file.

        Parameters
        ----------
        root_directory : str
            The root directory.
        file_name : str
            The file name.
        output_file : object
            The output file object.
        """
        output_file.write(f"## {file_name}\n\n")
        output_file.write("```python\n")
        with open(os.path.join(root_directory, file_name), "r") as file:
            file_content = file.read()
            output_file.write(file_content)
            print(file_content)  # Print the file content to the terminal
        output_file.write("\n```\n\n")

    def generate_quine(self) -> None:
        """
        Walks through the base directory and its subdirectories, excluding certain directories and file types,
        and writes the Python source code to a Markdown file with headings for each folder and subheadings for each file.
        The source code is enclosed in ```python ``` code blocks.
        """
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"quine_{current_time}.md"
        output_dir = Path().cwd() / "output"  # Output directory
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the output directory if it doesn't exist
        file_path = output_dir / file_name  # Output file path
        with open(file_path, "w") as output_file:
            for root_directory, directories, files in os.walk(self.base_directory):
                directories[:] = [
                    directory
                    for directory in directories
                    if directory not in self.excluded_directories
                ]
                if root_directory != ".":
                    output_file.write(f"# {os.path.relpath(root_directory, '..')}\n\n")
                for file_name in files:
                    if any(
                        file_name.endswith(extension)
                        for extension in self.included_extensions
                    ):
                        self.write_to_file(root_directory, file_name, output_file)


if __name__ == "__main__":
    # base_directory_in =r"C:\Users\jonma\github_repos\freemocap_organization\skelly_tracker\skelly_tracker\trackers"
    base_directory_in = r"C:\Users\jonma\github_repos\freemocap_organization\skelly_tracker\skelly_tracker\trackers"
    quine = Quine(
        base_directory=base_directory_in,
        excluded_directories=[
            "__pycache__",
            ".git",
            "output",
            # "mediapipe_tracker",
            # "charuco_tracker",
            "mmpose_tracker",
        ],
        included_extensions=[".py"],
    )
    quine.generate_quine()
