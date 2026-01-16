"""
Path Manager for HAMSTER path generation with retry and fallback logic.

This module provides a PathManager class that handles:
- Path generation with retry (max 2 attempts)
- Fallback to previous path or no-path image
- Tracking of path generation statistics for logging
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import time


class PathManager:
    """
    Manages path generation with retry and fallback strategies.

    Retry strategy:
    - Attempt path generation up to max_retries times
    - If all retries fail:
        - Use last successful path if available
        - Otherwise, use no-path (empty path)

    Initial path handling:
    - If frame 0 fails, use no-path image
    - Set initial_path when first successful path is generated
    """

    def __init__(
        self,
        vila_client,
        max_retries: int = 2,
        verbose: bool = False,
    ):
        """
        Initialize the PathManager.

        Args:
            vila_client: VILAClient instance for path generation
            max_retries: Maximum number of retry attempts (default: 2)
            verbose: Print detailed logs
        """
        self.vila_client = vila_client
        self.max_retries = max_retries
        self.verbose = verbose

        # State
        self.last_successful_path: Optional[List[Tuple[float, float, int]]] = None
        self.initial_path: Optional[List[Tuple[float, float, int]]] = None
        self.initial_path_set: bool = False

        # Statistics for current episode
        self.stats = self._init_stats()

    def _init_stats(self) -> Dict[str, Any]:
        """Initialize statistics dictionary."""
        return {
            "total_path_calls": 0,
            "path_successes": 0,
            "path_failures": 0,
            "retries": 0,
            "fallbacks_used": 0,
            "frame0_success": None,
            "vila_inference_times_ms": [],
        }

    def reset(self):
        """Reset state for new episode."""
        self.last_successful_path = None
        self.initial_path = None
        self.initial_path_set = False
        self.stats = self._init_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get current episode statistics."""
        return self.stats.copy()

    def generate_path(
        self,
        image: np.ndarray,
        task_instruction: str,
        is_frame0: bool = False,
    ) -> Tuple[Optional[List[Tuple[float, float, int]]], bool, bool]:
        """
        Generate path with retry and fallback logic.

        Args:
            image: RGB image array (H, W, 3)
            task_instruction: Task instruction for VILA
            is_frame0: Whether this is frame 0 of the episode

        Returns:
            Tuple of:
            - path: Generated path or fallback path (None if no-path)
            - success: Whether path generation succeeded (without fallback)
            - used_fallback: Whether a fallback was used
        """
        self.stats["total_path_calls"] += 1

        path = None
        success = False
        used_fallback = False
        total_retries = 0

        # Try path generation with retries
        for attempt in range(self.max_retries):
            start_time = time.time()

            path = self.vila_client.predict_path(
                image,
                task_instruction,
                verbose=self.verbose,
            )

            inference_time_ms = (time.time() - start_time) * 1000
            self.stats["vila_inference_times_ms"].append(inference_time_ms)

            if path is not None and len(path) > 0:
                success = True
                self.stats["path_successes"] += 1
                self.last_successful_path = path

                # Set initial path if not yet set
                if not self.initial_path_set:
                    self.initial_path = path
                    self.initial_path_set = True
                    if self.verbose:
                        print(f"[PathManager] Initial path set at attempt {attempt + 1}")

                break
            else:
                if attempt < self.max_retries - 1:
                    total_retries += 1
                    if self.verbose:
                        print(f"[PathManager] Retry {attempt + 1}/{self.max_retries - 1}")

        self.stats["retries"] += total_retries

        # Handle failure
        if not success:
            self.stats["path_failures"] += 1

            if self.last_successful_path is not None:
                # Use last successful path as fallback
                path = self.last_successful_path
                used_fallback = True
                self.stats["fallbacks_used"] += 1
                if self.verbose:
                    print(f"[PathManager] Using fallback (last successful path)")
            else:
                # No path available - return None (caller should use no-path image)
                path = None
                used_fallback = True
                self.stats["fallbacks_used"] += 1
                if self.verbose:
                    print(f"[PathManager] No fallback available, using no-path image")

        # Track frame 0 success
        if is_frame0:
            self.stats["frame0_success"] = success

        return path, success, used_fallback

    def get_initial_path(self) -> Optional[List[Tuple[float, float, int]]]:
        """
        Get the initial path for this episode.

        Returns:
            Initial path if set, None otherwise
        """
        return self.initial_path

    def has_initial_path(self) -> bool:
        """Check if initial path has been set."""
        return self.initial_path_set


class PathManagerOriginal:
    """
    Dummy PathManager for original mode (no VILA, no path generation).

    Provides the same interface as PathManager but returns no paths.
    Used for condition 1 and 4 (original ManiFlow without VILA).
    """

    def __init__(self):
        """Initialize dummy PathManager."""
        self.stats = self._init_stats()

    def _init_stats(self) -> Dict[str, Any]:
        """Initialize statistics dictionary."""
        return {
            "total_path_calls": 0,
            "path_successes": 0,
            "path_failures": 0,
            "retries": 0,
            "fallbacks_used": 0,
            "frame0_success": None,
            "vila_inference_times_ms": [],
        }

    def reset(self):
        """Reset state for new episode."""
        self.stats = self._init_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get current episode statistics (all zeros for original mode)."""
        return self.stats.copy()

    def generate_path(
        self,
        image: np.ndarray,
        task_instruction: str,
        is_frame0: bool = False,
    ) -> Tuple[None, bool, bool]:
        """
        Dummy path generation - always returns no path.

        Args:
            image: Ignored
            task_instruction: Ignored
            is_frame0: Ignored

        Returns:
            Tuple of (None, False, False)
        """
        return None, False, False

    def get_initial_path(self) -> None:
        """Always returns None for original mode."""
        return None

    def has_initial_path(self) -> bool:
        """Always returns False for original mode."""
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("PathManager Test")
    print("=" * 50)

    # Test PathManagerOriginal
    print("\n1. Testing PathManagerOriginal...")
    pm_original = PathManagerOriginal()

    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    path, success, fallback = pm_original.generate_path(dummy_image, "test")

    assert path is None
    assert success is False
    assert fallback is False
    print("   [PASS] PathManagerOriginal works correctly")

    # Test PathManager statistics
    print("\n2. Testing PathManager statistics reset...")

    class MockVILAClient:
        def __init__(self):
            self.call_count = 0

        def predict_path(self, image, instruction, verbose=False):
            self.call_count += 1
            # Fail on first call, succeed on second
            if self.call_count == 1:
                return None
            return [(0.5, 0.5, 0)]

    mock_client = MockVILAClient()
    pm = PathManager(mock_client, max_retries=2, verbose=True)

    # Generate path with retry
    path, success, fallback = pm.generate_path(dummy_image, "test", is_frame0=True)

    stats = pm.get_stats()
    print(f"   Stats: {stats}")

    assert stats["total_path_calls"] == 1
    assert stats["path_successes"] == 1
    assert stats["retries"] == 1
    assert stats["frame0_success"] is True
    print("   [PASS] PathManager statistics work correctly")

    # Test reset
    pm.reset()
    stats = pm.get_stats()
    assert stats["total_path_calls"] == 0
    print("   [PASS] PathManager reset works correctly")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
