"""Unit tests for ``region_for_iter`` (v2 Task 5 — multi-region round-robin).

These tests exercise the pure helper only — no JAX, no maps, no Modal.
"""
import pytest

from smoothride.rl.modal_train import region_for_iter


class TestRegionForIterCycles:
    """The function cycles through the region list in order."""

    def test_two_regions_alternates(self) -> None:
        regions = ["downtown", "mission"]
        assert region_for_iter(0, regions) == "downtown"
        assert region_for_iter(1, regions) == "mission"
        assert region_for_iter(2, regions) == "downtown"
        assert region_for_iter(3, regions) == "mission"

    def test_three_regions_cycles_correctly(self) -> None:
        regions = ["downtown", "nopa", "chinatown_fidi"]
        for i, expected in enumerate(regions * 3):
            assert region_for_iter(i, regions) == expected, (
                f"iter {i}: expected {expected!r}, got {region_for_iter(i, regions)!r}"
            )

    def test_large_iter_wraps(self) -> None:
        regions = ["downtown", "mission"]
        # iter 1000 % 2 == 0 → "downtown"
        assert region_for_iter(1000, regions) == "downtown"
        # iter 1001 % 2 == 1 → "mission"
        assert region_for_iter(1001, regions) == "mission"


class TestRegionForIterSingleRegion:
    """A single-element list always returns that region (backward compat)."""

    def test_single_region_constant(self) -> None:
        regions = ["downtown"]
        for it in range(10):
            assert region_for_iter(it, regions) == "downtown", (
                f"single-region: iter {it} should always return 'downtown'"
            )

    def test_single_region_large_iter(self) -> None:
        regions = ["nopa"]
        assert region_for_iter(9999, regions) == "nopa"


class TestRegionForIterEdgeCases:
    """Edge cases: empty list raises, iter 0 returns first element."""

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            region_for_iter(0, [])

    def test_iter_zero_returns_first(self) -> None:
        regions = ["chinatown_fidi", "nopa", "mission", "downtown"]
        assert region_for_iter(0, regions) == "chinatown_fidi"

    def test_all_sf_regions(self) -> None:
        """Smoke: all four SF region keys can be used without error."""
        regions = ["downtown", "mission", "nopa", "chinatown_fidi"]
        results = [region_for_iter(i, regions) for i in range(8)]
        assert results == regions + regions
