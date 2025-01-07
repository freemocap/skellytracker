from skellytracker.system.default_paths import get_log_file_path
from skellytracker.system.logging_configuration import configure_logging
configure_logging(log_file_path=str(get_log_file_path()))
