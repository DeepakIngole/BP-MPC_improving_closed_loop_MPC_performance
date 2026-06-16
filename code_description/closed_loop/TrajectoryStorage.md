**Tags:** #class

## Overview

The `TrajectoryStorage` class is a dictionary-like container (inheriting from `UserDict`) specifically tailored for aggregating and archiving simulation rollout data.

## Purpose

After running a simulation, the resulting data is usually a mix of massive, compiled JAX arrays and simple native Python metadata (like configuration parameters or runtimes). `TrajectoryStorage` seamlessly bridges this gap by providing a single `.save()` method that automatically partitions the data by type, ensuring everything is serialized safely and efficiently.

## Key Methods

- **`save(directory)`**: Iterates through the stored data, converts any JAX arrays into standard NumPy arrays, and compresses them into a single `trajectories.npz` file. Simultaneously, it extracts any basic Python types (scalars, strings) and writes them to a `metadata.json` file in the same directory.

## Dependencies

_This class is a pure utility leaf node and has no internal dependencies on other core framework classes. It relies only on standard Python libraries, `numpy`, and `jax`._