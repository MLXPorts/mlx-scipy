import mlx.core as mx
from scipy._lib._util import DecimalNumber, IntNumber


def _cy_wrapper_centered_discrepancy(
        sample: mx.array,
        iterative: bool, 
        workers: IntNumber,
) -> float: ...


def _cy_wrapper_wrap_around_discrepancy(
        sample: mx.array,
        iterative: bool, 
        workers: IntNumber,
) -> float: ...


def _cy_wrapper_mixture_discrepancy(
        sample: mx.array,
        iterative: bool, 
        workers: IntNumber,
) -> float: ...


def _cy_wrapper_l2_star_discrepancy(
        sample: mx.array,
        iterative: bool,
        workers: IntNumber,
) -> float: ...


def _cy_wrapper_update_discrepancy(
        x_new_view: mx.array,
        sample_view: mx.array,
        initial_disc: DecimalNumber,
) -> float: ...


def _cy_van_der_corput(
        n: IntNumber,
        base: IntNumber,
        start_index: IntNumber,
        workers: IntNumber,
) -> mx.array: ...


def _cy_van_der_corput_scrambled(
        n: IntNumber,
        base: IntNumber,
        start_index: IntNumber,
        permutations: mx.array,
        workers: IntNumber,
) -> mx.array: ...
