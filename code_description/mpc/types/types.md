**Tags:** #file

## Overview

The `types.py` module is a centralized repository for the internal data types, type aliases, and NamedTuples shared across the entire MPC package.

## Purpose

By grouping foundational data structures into a single file, it completely eliminates the risk of circular imports within the framework. It defines the strict type contracts (like `ArrayIn` for parameter dictionaries and `QArray` for dense/sparse arrays) used by the assembler, the solvers, and the user-facing descriptors.

## Dependencies

- Contains the definitions for [[QPData]] and [[SlackData]].