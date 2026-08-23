"""Top-level package for skellytracker"""

__package_name__ = "skellytracker"
__version__ = "v2024.09.1019"

__author__ = """Skelly FreeMoCap"""
__email__ = "info@freemocap.org"
__repo_owner_github_user_name__ = "freemocap"
__repo_url__ = (
    f"https://github.com/{__repo_owner_github_user_name__}/{__package_name__}"
)
__repo_issues_url__ = f"{__repo_url__}/issues"

# ruff: noqa: F401, E402

from beartype.claw import beartype_this_package
beartype_this_package()
