"""
Profiling utilities for performance analysis.

This module provides easy-to-use decorators and functions for profiling
decline curve analysis code using line_profiler.

Usage:
    # Decorate functions you want to profile
    from decline_analysis.profiling import profile

    @profile
    def my_function():
        # Your code here
        pass

    # Or use context manager
    with profile_context("My operation"):
        # Your code here
        pass
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

try:
    from line_profiler import LineProfiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    LineProfiler = None


# Global profiler instance
_profiler = None


def get_profiler() -> Optional[LineProfiler]:
    """Get or create the global profiler instance."""
    global _profiler
    if LINE_PROFILER_AVAILABLE and _profiler is None:
        _profiler = LineProfiler()
    return _profiler


def profile(func: Callable) -> Callable:
    """
    Decorator to profile a function with line_profiler.

    If line_profiler is not installed, this decorator does nothing.

    Args:
        func: Function to profile

    Returns:
        Wrapped function

    Example:
        >>> from decline_analysis.profiling import profile
        >>>
        >>> @profile
        >>> def compute_forecast(data):
        >>>     # Your code here
        >>>     pass
        >>>
        >>> # Run your function
        >>> compute_forecast(data)
        >>>
        >>> # Print profiling results
        >>> from decline_analysis.profiling import print_stats
        >>> print_stats()
    """
    if not LINE_PROFILER_AVAILABLE:
        # Return function unchanged if profiler not available
        return func

    profiler = get_profiler()
    if profiler is not None:
        profiler.add_function(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def print_stats():
    """
    Print profiling statistics for all profiled functions.

    Example:
        >>> from decline_analysis.profiling import print_stats
        >>> # After running profiled functions
        >>> print_stats()
    """
    profiler = get_profiler()
    if profiler is not None:
        profiler.print_stats()
    else:
        print("line_profiler not available. Install with: pip install line_profiler")


def save_stats(filename: str):
    """
    Save profiling statistics to a file.

    Args:
        filename: Output file path

    Example:
        >>> from decline_analysis.profiling import save_stats
        >>> save_stats("profiling_results.txt")
    """
    profiler = get_profiler()
    if profiler is not None:
        with open(filename, "w") as f:
            profiler.print_stats(stream=f)
        print(f"Profiling results saved to {filename}")
    else:
        print("line_profiler not available. Install with: pip install line_profiler")


@contextmanager
def profile_context(name: str = "Operation", print_time: bool = True):
    """
    Context manager to time a block of code.

    Args:
        name: Name of the operation being profiled
        print_time: Whether to print elapsed time

    Yields:
        Dictionary with timing information

    Example:
        >>> from decline_analysis.profiling import profile_context
        >>>
        >>> with profile_context("Forecasting 100 wells") as timer:
        >>>     for well in wells:
        >>>         forecast(well)
        >>>
        >>> print(f"Took {timer['elapsed']:.2f} seconds")
    """
    timer = {"start": time.time(), "elapsed": 0}

    if print_time:
        print(f"\n{'='*60}")
        print(f"Starting: {name}")
        print(f"{'='*60}")

    try:
        yield timer
    finally:
        timer["elapsed"] = time.time() - timer["start"]
        if print_time:
            print(f"{'='*60}")
            print(f"Completed: {name}")
            print(f"Elapsed time: {timer['elapsed']:.3f} seconds")
            print(f"{'='*60}\n")


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints execution time

    Example:
        >>> from decline_analysis.profiling import time_function
        >>>
        >>> @time_function
        >>> def slow_operation():
        >>>     # Your code here
        >>>     pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f} seconds")
        return result

    return wrapper


def reset_profiler():
    """Reset the global profiler to start fresh profiling."""
    global _profiler
    if LINE_PROFILER_AVAILABLE:
        _profiler = LineProfiler()
    else:
        _profiler = None


# Example profiling configuration
PROFILING_ENABLED = False  # Set to True to enable profiling by default


def conditional_profile(func: Callable) -> Callable:
    """
    Conditionally profile based on PROFILING_ENABLED flag.

    This is useful for leaving profiling decorators in production code
    without performance overhead.

    Args:
        func: Function to conditionally profile

    Returns:
        Wrapped function
    """
    if PROFILING_ENABLED:
        return profile(func)
    return func
