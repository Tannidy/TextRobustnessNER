import os
import pathlib
import zipfile

from .logger import logger

# Hide an error message from `tokenizers` if this process is forked.
os.environ["TOKENIZERS_PARALLELISM"] = "True"


def path_in_cache(file_path):
    try:
        os.makedirs(TEXTROBUSTNESS_CACHE_DIR)
    except FileExistsError:  # cache path exists
        pass
    return os.path.join(TEXTROBUSTNESS_CACHE_DIR, file_path)


def unzip_file(path_to_zip_file, unzipped_folder_path):
    """Unzips a .zip file to folder path."""
    logger.info(f"Unzipping file {path_to_zip_file} to {unzipped_folder_path}.")
    enclosing_unzipped_path = pathlib.Path(unzipped_folder_path).parent
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(enclosing_unzipped_path)


def set_cache_dir(cache_dir):
    """Sets all relevant cache directories to ``TA_CACHE_DIR``."""
    # Tensorflow Hub cache directory
    os.environ["TFHUB_CACHE_DIR"] = cache_dir
    # HuggingFace `transformers` cache directory
    os.environ["PYTORCH_TRANSFORMERS_CACHE"] = cache_dir
    # HuggingFace `datasets` cache directory
    os.environ["HF_HOME"] = cache_dir
    # Basic directory for Linux user-specific non-data files
    os.environ["XDG_CACHE_HOME"] = cache_dir


TEXTROBUSTNESS_CACHE_DIR = os.environ.get(
    "TA_CACHE_DIR", os.path.expanduser("~/.cache/TextRobustness")
)
if "TA_CACHE_DIR" in os.environ:
    set_cache_dir(os.environ["TA_CACHE_DIR"])

