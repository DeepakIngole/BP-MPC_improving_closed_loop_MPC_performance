from typing import Any
import numpy as np

class RunLogger:
    """A dynamic terminal logger that formats runtime metrics into a clean table.
    Automatically handles column spacing and reprints headers when fields change.
    """
    
    def __init__(self, padding: int = 4):
        self.padding = padding
        self._last_keys = None
        self._col_widths = {}

    def log(self, **metrics: Any) -> None:
        """Logs a row of metrics. Pass metrics as kwargs, e.g., log(iter=1, cost=0.5)"""
        current_keys = list(metrics.keys())
        
        # If the fields changed (or this is the first run), rebuild the header
        if current_keys != self._last_keys:
            self._update_layout(metrics)
            self._print_header()
            self._last_keys = current_keys
            
        self._print_row(metrics)

    def _update_layout(self, metrics: dict) -> None:
        """Calculates the required width for each column based on header names."""
        self._col_widths = {
            key: max(len(str(key)), 10) + self.padding 
            for key in metrics.keys()
        }

    def _print_header(self) -> None:
        """Prints the table header and a separator line."""
        header_cols = [f"{key:^{self._col_widths[key]}}" for key in self._last_keys or self._col_widths.keys()]
        header_str = "|" + "|".join(header_cols) + "|"
        
        separator = "+" + "+".join("-" * self._col_widths[key] for key in self._col_widths.keys()) + "+"
        
        print("\n" + separator)
        print(header_str)
        print(separator)

    def _print_row(self, metrics: dict) -> None:
        """Formats and prints a single row of data."""
        row_cols = []
        for key, val in metrics.items():
            width = self._col_widths[key]
            
            # Format based on type
            if isinstance(val, (int, np.integer)):
                formatted_val = f"{val:d}"
            elif isinstance(val, (float, np.floating)):
                # Use scientific notation for floats to ensure they fit cleanly
                formatted_val = f"{val:.4e}"
            else:
                formatted_val = str(val)[:width-2] # Truncate string if too long
                
            # Right-align values for clean numerical reading
            row_cols.append(f"{formatted_val:>{width - 1}} ")
            
        row_str = "|" + "|".join(row_cols) + "|"
        print(row_str)