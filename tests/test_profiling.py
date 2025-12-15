"""Tests for profiling utilities."""

import time

import pytest

from decline_curve.profiling import (
    get_profiler,
    print_stats,
    profile,
    profile_context,
    reset_profiler,
)


class TestProfiling:
    """Test profiling decorators and utilities."""

    def test_profile_decorator(self):
        """Test @profile decorator."""

        @profile
        def test_function(x):
            """Test function for profiling."""
            time.sleep(0.01)  # Small delay
            return x * 2

        # Function should still work normally
        result = test_function(5)
        assert result == 10

    def test_profile_context_manager(self):
        """Test profile_context context manager."""
        with profile_context("test_operation"):
            time.sleep(0.01)
            result = 2 + 2

        assert result == 4

    def test_profile_nested(self):
        """Test nested profiling."""

        @profile
        def inner_func(x):
            return x + 1

        @profile
        def outer_func(x):
            return inner_func(x) * 2

        result = outer_func(5)
        assert result == 12

    def test_get_profiler(self):
        """Test get_profiler function."""
        profiler = get_profiler()
        # Should return None or LineProfiler instance depending on availability
        assert profiler is None or hasattr(profiler, "enable_by_count")

    def test_reset_profiler(self):
        """Test reset_profiler function."""
        reset_profiler()
        # Should not raise an error
        profiler = get_profiler()
        assert profiler is None or hasattr(profiler, "enable_by_count")

    def test_print_stats(self, capsys):
        """Test print_stats function."""

        @profile
        def test_func(x):
            return x * 2

        test_func(5)
        print_stats()

        # Should print something or nothing (depending on line_profiler availability)
        captured = capsys.readouterr()
        # Function should complete without error

    def test_profiling_without_line_profiler(self):
        """Test that profiling works even without line_profiler."""

        # Decorator should still work even if line_profiler is not available
        # (it will just return the function unchanged)
        @profile
        def test_func(x):
            return x * 3

        result = test_func(7)
        assert result == 21


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
