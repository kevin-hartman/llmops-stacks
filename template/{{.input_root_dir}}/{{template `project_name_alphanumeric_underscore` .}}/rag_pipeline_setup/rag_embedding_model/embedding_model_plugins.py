from os.path import abspath, join, isfile, split, dirname, exists
from os import listdir
from abc import ABC, abstractmethod
import logging
from importlib import util

from rag_pipeline_setup.consts import FILE_SKIP_LIST, DEFAULT_DIRECTORY

class EmbeddingModelPlugins(ABC):
    embedding_model_plugins = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "Abstract" not in cls.__name__:
            print(f"Registering Embedding model plugin '{cls.__name__}' with flavor '{cls.embedding_model_flavor()}'")
            cls.embedding_model_plugins[cls.embedding_model_flavor()] = cls

    @staticmethod
    @abstractmethod
    def embedding_model_flavor() -> str:
        pass


class EmbeddingModelPluginsDiscovery():
    def __init__(self):
        self._setup_logger()

    def _setup_logger(self) -> None:
        """
        Create a class-specific logger with a unique name.
        """
        # Use the class name and a unique identifier (alias) to create a distinct logger
        logger_name = f"{self.__class__.__name__}"
        self._logger = logging.getLogger(logger_name)

        # Only add handler if no handlers exist to prevent duplicate logging
        if not self._logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self._logger.addHandler(console_handler)

            # Set default logging level to INFO
            self._logger.setLevel(logging.INFO)

    @staticmethod
    def construct_path(plugins_dir: str = DEFAULT_DIRECTORY,
                       abs_path: bool = False) -> str:
        if abs_path:
            return plugins_dir
        current_file_path = dirname(abspath(__file__))
        full_plugin_path = join(current_file_path, plugins_dir)
        return full_plugin_path

    def list_files_in_plugin_dir(self, plugins_directory_path: str) -> list[str]:
        files = [file for file in listdir(plugins_directory_path)
                 if isfile(join(plugins_directory_path, file))
                 and file not in FILE_SKIP_LIST
                 and file.endswith(".py")]

        files = [join(plugins_directory_path, file) for file in files]
        self._logger.info(f"List of Embedding model plugins found: '{files}'")
        if len(files) == 0:
            self._logger.error(f"No Embedding model plugins found in directory: '{plugins_directory_path}'")
            raise RuntimeError(f"Empty plugin directory: '{plugins_directory_path}'")

        return files

    def load_plugins(self, plugin_file_list: list[str]) -> None:
        for file in plugin_file_list:
            try:
                self._logger.info(f"Attempting to load Embedding model plugin from file: '{file}'")
                if not exists(file):
                    self._logger.error(f"File for plugin at location '{file}' not found")
                    raise FileNotFoundError(f"Missing file at location: '{file}'")
                module_name, _ = split(file)
                spec = util.spec_from_file_location(module_name, file)
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._logger.info(f"Successfully loaded Embedding Model plugin from file: '{file}'")
            except Exception:
                self._logger.error(f"Error loading Embedding model plugin module: '{file}'. Please check trace log")
                raise

    def discover_embedding_model_plugins(self, plugins_dir: str = DEFAULT_DIRECTORY,
                                         abs_path: bool = False) -> None:
        plugins_directory_path = self.construct_path(plugins_dir, abs_path)
        self._logger.info(f"Finding Embedding model plugins from directory: '{plugins_directory_path}'")
        valid_plugin_files = self.list_files_in_plugin_dir(plugins_directory_path)
        self.load_plugins(valid_plugin_files)
