"""
Simple test to verify the MLX cosine distribution implementation.

This tests the _cosine_mlx module which replaces the C implementation
in _cosine.c with pure MLX code.
"""

import mlx.core as mx
from scipy_mlx.special._cosine_mlx import cosine_cdf, cosine_invcdf

def test_cosine_cdf_basic():
    """Test basic CDF functionality"""
    print("Testing cosine_cdf...")

    # Test at key points
    test_points = mx.array([
        -mx.pi,      # Should be ~0
        mx.array(0.0, dtype=mx.float32),  # Should be 0.5
        mx.pi,       # Should be ~1
    ], dtype=mx.float32)

    results = cosine_cdf(test_points)
    print(f"  x = {test_points}")
    print(f"  CDF(x) = {results}")

    # Check values are in [0, 1]
    assert mx.all(mx.greater_equal(results, mx.array(0.0, dtype=mx.float32))), "CDF should be >= 0"
    assert mx.all(mx.less_equal(results, mx.array(1.0, dtype=mx.float32))), "CDF should be <= 1"

    print("  ✓ CDF values are in [0, 1]")

    # Check monotonicity
    x_mono = mx.array(mx.linspace(-3.0, 3.0, 100), dtype=mx.float32)
    cdf_mono = cosine_cdf(x_mono)
    diffs = cdf_mono[1:] - cdf_mono[:-1]
    assert mx.all(mx.greater_equal(diffs, mx.array(0.0, dtype=mx.float32))), "CDF should be monotonic"

    print("  ✓ CDF is monotonically increasing")


def test_cosine_invcdf_basic():
    """Test basic inverse CDF functionality"""
    print("\nTesting cosine_invcdf...")

    # Test at key probabilities
    test_probs = mx.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=mx.float32)

    results = cosine_invcdf(test_probs)
    print(f"  p = {test_probs}")
    print(f"  InvCDF(p) = {results}")

    # Check p=0 gives -π
    assert mx.abs(results[0] - (-mx.pi)) < 1e-5, f"InvCDF(0) should be -π, got {results[0]}"
    print(f"  ✓ InvCDF(0) = {results[0]:.6f} ≈ -π")

    # Check p=1 gives π
    assert mx.abs(results[4] - mx.pi) < 1e-5, f"InvCDF(1) should be π, got {results[4]}"
    print(f"  ✓ InvCDF(1) = {results[4]:.6f} ≈ π")

    # Check p=0.5 gives 0
    assert mx.abs(results[2]) < 1e-4, f"InvCDF(0.5) should be ~0, got {results[2]}"
    print(f"  ✓ InvCDF(0.5) = {results[2]:.6f} ≈ 0")


def test_roundtrip():
    """Test that InvCDF(CDF(x)) ≈ x"""
    print("\nTesting CDF/InvCDF roundtrip...")

    # Test points in valid range
    x_test = mx.array(mx.linspace(-2.5, 2.5, 20), dtype=mx.float32)

    # Forward: x → p
    p = cosine_cdf(x_test)

    # Backward: p → x'
    x_reconstructed = cosine_invcdf(p)

    # Check roundtrip accuracy
    errors = mx.abs(x_reconstructed - x_test)
    max_error = mx.max(errors)
    mean_error = mx.mean(errors)

    print(f"  Max roundtrip error: {max_error:.2e}")
    print(f"  Mean roundtrip error: {mean_error:.2e}")

    assert max_error < 1e-5, f"Max roundtrip error too large: {max_error}"
    print("  ✓ Roundtrip accuracy within tolerance")


def test_invalid_inputs():
    """Test handling of invalid inputs"""
    print("\nTesting invalid input handling...")

    # Test invcdf with p outside [0, 1]
    invalid_p = mx.array([-0.1, 1.5], dtype=mx.float32)
    results = cosine_invcdf(invalid_p)

    # Should return NaN for invalid inputs
    assert mx.all(mx.isnan(results)), "Invalid probabilities should return NaN"
    print("  ✓ Invalid probabilities return NaN")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MLX Cosine Distribution Implementation")
    print("=" * 60)

    try:
        test_cosine_cdf_basic()
        test_cosine_invcdf_basic()
        test_roundtrip()
        test_invalid_inputs()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
