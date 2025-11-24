import numpy as np

def in_inside_caloremeter(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """returns a boolean mask for particles inside the calorimeter acceptance"""
    r = np.sqrt(x**2 + y**2)
    return (r > 1400) | (np.abs(z) > 3000)