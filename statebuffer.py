import warnings
from typing import Generator, Optional, Union

import numpy as np
import torch as th
import io
import json
import os
import pathlib
import pickle
from typing import Any, Dict, Optional, Tuple, Union

class StateBuffer(object):

    def __init__(
        self,
        d_type,
        obs_shape,
        buffer_size: int,
        device: Union[th.device, str] = "cuda",
    ):
        super(StateBuffer, self).__init__()
        self.obs_shape = obs_shape
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.observations = np.zeros((self.buffer_size, ) + self.obs_shape, dtype=d_type)

    def add(self, obs:np.ndarray) -> None:
        self.observations[self.pos] = np.array(obs).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)  
        return th.as_tensor(self.observations[batch_inds,:]).to(self.device)
              

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False    


def open_path(path: Union[str, pathlib.Path, io.BufferedIOBase], mode: str, verbose: int = 0, suffix: Optional[str] = None):
    """
    Opens a path for reading or writing with a preferred suffix and raises debug information.
    If the provided path is a derivative of io.BufferedIOBase it ensures that the file
    matches the provided mode, i.e. If the mode is read ("r", "read") it checks that the path is readable.
    If the mode is write ("w", "write") it checks that the file is writable.
    If the provided path is a string or a pathlib.Path, it ensures that it exists. If the mode is "read"
    it checks that it exists, if it doesn't exist it attempts to read path.suffix if a suffix is provided.
    If the mode is "write" and the path does not exist, it creates all the parent folders. If the path
    points to a folder, it changes the path to path_2. If the path already exists and verbose == 2,
    it raises a warning.
    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param mode: how to open the file. "w"|"write" for writing, "r"|"read" for reading.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    if not isinstance(path, io.BufferedIOBase):
        raise TypeError("Path parameter has invalid type.", io.BufferedIOBase)
    if path.closed:
        raise ValueError("File stream is closed.")
    mode = mode.lower()
    try:
        mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
    except KeyError:
        raise ValueError("Expected mode to be either 'w' or 'r'.")
    if ("w" == mode) and not path.writable() or ("r" == mode) and not path.readable():
        e1 = "writable" if "w" == mode else "readable"
        raise ValueError(f"Expected a {e1} file.")
    return path

def load_from_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase], verbose: int = 0) -> Any:
    """
    Load an object from the path. If a suffix is provided in the path, it will use that suffix.
    If the path does not exist, it will attempt to load using the .pkl suffix.
    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    """
    with open_path(path, "r", verbose=verbose, suffix="pkl") as file_handler:
        return pickle.load(file_handler)

def save_to_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase], obj: Any, verbose: int = 0) -> None:
    """
    Save an object to path creating the necessary folders along the way.
    If the path exists and is a directory, it will raise a warning and rename the path.
    If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.
    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param obj: The object to save.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    """
    with open_path(path, "w", verbose=verbose, suffix="pkl") as file_handler:
        pickle.dump(obj, file_handler)