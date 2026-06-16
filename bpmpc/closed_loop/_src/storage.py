import json
from pathlib import Path
from collections import UserDict
from typing import Union, Any

import numpy as np
import jax

class TrajectoryStorage(UserDict):
    """A dictionary-like container for trajectories and parameters 
    that can easily serialize its contents to a directory.
    """
    
    def save(self, directory: Union[str, Path]) -> None:
        """Saves the stored data to the specified directory.
        
        Arrays are compressed into a single .npz file, while native 
        Python types (scalars, strings) are saved to a metadata.json file.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        arrays = {}
        metadata = {}
        
        for key, value in self.data.items():
            # Convert JAX arrays to NumPy arrays for safe storage
            if isinstance(value, (np.ndarray, jax.Array)):
                arrays[key] = np.asarray(value)
            elif isinstance(value, (list, tuple)) and any(isinstance(v, (np.ndarray, jax.Array)) for v in value):
                 # Convert lists/tuples of arrays into stacked arrays
                 arrays[key] = np.asarray(value)
            else:
                # Store basic Python types in metadata
                metadata[key] = value
                
        # Save arrays efficiently
        if arrays:
            np.savez_compressed(dir_path / "trajectories.npz", **arrays)
            
        # Save metadata as JSON
        if metadata:
            with open(dir_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)