"""Utility functions for displaying algorithm information."""

from __future__ import annotations

import locale
import sys
import time
from typing import Mapping, Union


def _supports_box_drawing() -> bool:
    """
    Check if the terminal supports boxed characters.

    Returns
    -------
    bool
        True if the terminal likely supports boxed characters, False otherwise.
    """
    enc = (sys.stdout.encoding or locale.getpreferredencoding(False)).lower()
    if not enc.startswith("utf"):
        return False
    try:
        "┌".encode(enc)
    except UnicodeEncodeError:
        return False
    return True


class TableFormatter:
    """
    Format tabular data into strings for display in the console.

    Parameters
    ----------
    headers : Mapping[str, int]
        Mapping of column names to their respective widths, in the format
        {column_name: column_width}.
    """

    def __init__(self, headers: Mapping[str, int]) -> None:
        if not headers or not isinstance(headers, Mapping):
            raise ValueError("'headers' must be a non-empty dictionary.")
        self.headers: Mapping[str, int] = headers
        self._ascii_only = not _supports_box_drawing()

    def _border(self, left: str, middle: str, right: str, line: str, new_line: bool = True) -> str:
        """
        Create a horizontal border for the table.

        Parameters
        ----------
        left: str
            Character on the left side of the border.
        middle: str
            Character separator between columns.
        right: str
            Character on the right side of the border.
        line: str
            Character used to fill the border.
        new_line: bool, optional
            If True, adds a line break before the border (default is True).

        Returns
        -------
        str
            String representing the horizontal border.
        """
        nl = '\n' if new_line else ''
        return f"{nl}{left}{middle.join(line * w for w in self.headers.values())}{right}"

    def get_header(self):
        """
        Generate the table header, including the top border, column headings, and separator line.

        Returns
        -------
        str
            Formatted string of the table header.
        """
        if self._ascii_only:
            top = self._border("+", "+", "+", "-")
            sep = self._border("+", "+", "+", "-")
            cell_border = "|"
        else:
            top = self._border("┌", "┬", "┐", "─")
            sep = self._border("├", "┼", "┤", "─")
            cell_border = "│"

        titles = '\n' + cell_border + cell_border.join(
            f"{h:^{self.headers[h]}}" for h in self.headers
        ) + cell_border

        return top + titles + sep

    def get_row(self, values: Mapping[str, Union[str, int, float]]):
        """
        Generate a formatted row for the table data.

        Parameters
        ----------
        values : Mapping[str, Union[str, int, float]]
            Dictionary with values for each column, in the format
            {column_name: value}.

        Returns
        -------
        str
        Formatted string of the table row.
        """
        border = "|" if self._ascii_only else "│"
        row = border + border.join(
            f"{values.get(h, ''):^{self.headers[h]}}" for h in self.headers
        ) + border

        return row

    def get_bottom(self, new_line: bool = False):
        """
        Generate the table's bottom border.

        Parameters
        ----------
        new_line : bool, Optional
            If True, adds a line break before the border (default is False).

        Returns
        -------
        str
            Formatted string for the bottom border.
        """
        if self._ascii_only:
            bottom = self._border("+", "+", "+", "-", new_line)
        else:
            bottom = self._border("└", "┴", "┘", "─", new_line)
        return bottom


class ProgressTable(TableFormatter):
    """
    Display a formatted table in the console to track the algorithm's progress.

    Parameters
    ----------
    headers : Mapping[str, int]
        Mapping {column_name: column_width}.
    verbose : bool, default=True
        If False, prints nothing to the terminal.
    """

    def __init__(self, headers: Mapping[str, int], verbose: bool = True) -> None:
        super().__init__(headers)
        if not headers or not isinstance(headers, Mapping):
            raise ValueError("O parâmetro 'headers' deve ser um dicionario não vazio.")
        self.verbose: bool = verbose
        self.headers: Mapping[str, int] = headers
        self._ascii_only = not _supports_box_drawing()
        if self.verbose:
            self._print_header()
            self._start = time.perf_counter()

    def _print_header(self) -> None:
        """Print the table header."""
        print(self.get_header())

    def update(self, values: Mapping[str, Union[str, int, float]]) -> None:
        """
        Add a new row of values to the table.

        Parameters
        ----------
        values: Mapping[str, Union[str, int, float]]
            Keys must match the columns defined in headers.
        """
        if not self.verbose:
            return
        print(self.get_row(values))

    def finish(self) -> None:
        """End the table display, printing the bottom border and total time."""
        if not self.verbose:
            return

        print(self.get_bottom())
        print(f"Total time: {time.perf_counter() - self._start:.6f} seconds")
