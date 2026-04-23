import jax
import numpy as np
import jax.numpy as jnp
from jaxlib import xla_client
from jax.extend.core import Primitive
from jax.interpreters import ad, mlir
from jax._src.interpreters.mlir import ir, custom_call
from jaxlib.mlir.dialects import mhlo

from harmonica.core import bindings

# Enable double floating precision.
jax.config.update("jax_enable_x64", True)


def ir_dtype(np_dtype):
    """Convert a NumPy dtype or Python type to an MLIR IR type."""
    np_dtype = np.dtype(np_dtype)
    if np_dtype == np.float32:
        return ir.F32Type.get()
    elif np_dtype == np.float64:
        return ir.F64Type.get()
    elif np_dtype == np.int32:
        return ir.IntegerType.get_signless(32)
    elif np_dtype == np.int64:
        return ir.IntegerType.get_signless(64)
    else:
        raise TypeError(f"Unsupported dtype: {np_dtype}")


def ir_constant(val):
    if not isinstance(val, np.generic):
        dtype = np.dtype(type(val))
    else:
        dtype = val.dtype
    arr = np.array(val, dtype=dtype).reshape(())
    attr = ir.DenseElementsAttr.get(arr)
    return mhlo.ConstantOp(attr).result


def _harmonica_transit_common(primitive_fn, times, params, r):
    """
    Internal helper to evaluate JAX-based Harmonica with broadcasted inputs.

    Parameters
    ----------
    primitive_fn : Callable
        JAX primitive function that returns (flux, Jacobian).
    times : array_like
        Scalar or 1D array of times [days] at which to evaluate the model.
    params : list of scalars or array_like
        Orbital and limb-darkening parameters to be broadcast.
    r : array_like
        Ripple coefficients specifying the planet silhouette. Must be shape
        (k,) or (n, k).

    Returns
    -------
    outputs : tuple
        Tuple (flux, [Jacobian]) with NaNs and Infs sanitized.
    """
    MIN_ABS = 1e-9                 # Minimum allowed coefficient to avoid bugs
    SIGMOID_WIDTH = 0.1 * MIN_ABS  # Width of sigmoid transition to MIN_ABS
    SAFE_SIGN_EPS = 1e-12          # Jitter to avoid divide-by-zero in sign

    def smooth_min_abs(x, min_abs=MIN_ABS, softness=SIGMOID_WIDTH):
        """Enforce differentiable minimum absolute value using a sigmoid blend.

        This avoids non-differentiable behavior around |x| ≈ 0 by replacing it
        with a smooth transition to `min_abs`.

        Parameters
        ----------
        x : float or array_like
            Input value(s) to regularize.
        min_abs : float
            Minimum allowed magnitude.
        softness : float
            Width of sigmoid transition between original and floored value.

        Returns
        -------
        output : ndarray
            Regularized values with smooth minimum magnitude.
        """
        abs_x = jnp.abs(x)
        scale = (abs_x - min_abs) / softness
        blend = jax.nn.sigmoid(scale)
        # Use a differentiable approximation to sign(x)
        safe_sign = x / jnp.sqrt(x**2 + SAFE_SIGN_EPS)
        return blend * x + (1 - blend) * min_abs * safe_sign

    def ensure_last_two_nonzero(r, min_tail=MIN_ABS,
                                softness=SIGMOID_WIDTH):
        """Ensure the last two ripple coefficients are not simultaneously zero.

        Adds a soft floor to r[-2:] via sigmoid blending to avoid degeneracies.

        Parameters
        ----------
        r : array_like
            Ripple vector (length ≥ 3), shaped (k,).
        min_tail : float
            Minimum allowed absolute value for tail coefficients.
        softness : float
            Width of the soft transition region for sigmoid.

        Returns
        -------
        r_fixed : ndarray
            Ripple vector with non-zero tail values enforced.
        """
        tail = r[-2:]
        abs_tail = jnp.abs(tail)
        blend = jax.nn.sigmoid((jnp.sum(abs_tail) - min_tail) / softness)
        safe_tail = jnp.sign(tail + SAFE_SIGN_EPS) * min_tail
        new_tail = blend * tail + (1 - blend) * safe_tail
        return r.at[-2:].set(new_tail)

    # Validate times
    times = jnp.atleast_1d(jnp.asarray(times))
    if times.ndim > 1:
        raise ValueError("`times` must be a scalar or 1D array")
    n = times.shape[0]

    # Broadcast scalar/1D params to shape (n,)
    broadcasted_params = []
    for p in params:
        p = jnp.asarray(p, dtype=jnp.float64)
        if p.ndim == 0 or (p.ndim == 1 and p.shape[0] != n):
            p = jnp.broadcast_to(p, (n,))
        elif p.ndim != 1:
            raise ValueError(f"Unexpected parameter shape: {p.shape}")
        broadcasted_params.append(p)

    # Regularize ripple input
    r = jnp.asarray(r, dtype=jnp.float64)

    if r.ndim == 1:
        # Ensure ripple vector has odd length by appending a 0 if needed.
        if r.shape[0] % 2 == 0:
            r = jnp.append(r, 0.0)

        # Track whether we need to flip the transit (i.e., r[0] < 0).
        flip = r[0] < 0

        # Smoothly enforce a minimum magnitude on r[0] while preserving sign,
        # to avoid degeneracies with near-zero transit depths.
        r0_fixed = smooth_min_abs(jnp.abs(r[0]))
        r = r.at[0].set(r0_fixed)

        # Smoothly enforce that the last two coefficients are not
        # simultaneously 0, which could lead to numerical issues in shape
        # evaluation.
        r = ensure_last_two_nonzero(r)
    elif r.ndim == 2:
        # Ensure ripple vectors each have odd length (axis 1), pad last column
        # with 0.
        if r.shape[1] % 2 == 0:
            r = jnp.pad(r, ((0, 0), (0, 1)), constant_values=0.0)

        # Flip vector-wise: record where r[0] < 0 for each row.
        flip = r[:, 0] < 0

        # Enforce a minimum |r[0]| smoothly, preserving sign.
        r0_fixed = smooth_min_abs(jnp.abs(r[:, 0]))
        r = r.at[:, 0].set(r0_fixed)

        # For each time step, enforce that last two Fourier terms are non-zero.
        r = jax.vmap(ensure_last_two_nonzero)(r)
    else:
        raise ValueError(f"`r` must be shape (k,) or (n, k); got {r.shape}")

    # Split ripple into list of args
    if r.ndim == 1:
        r_list = [jnp.broadcast_to(r[i], (n,)) for i in range(r.shape[0])]
    else:
        r_list = [r[:, i] for i in range(r.shape[1])]

    # Combine all args
    args = [jnp.asarray(arg, dtype=jnp.float64)
            for arg in (times, *broadcasted_params, *r_list)]

    flux, *rest = primitive_fn(*args)

    # Sanitize flux and Jacobian to avoid NaNs/Infs from C++ backend
    flux = jnp.nan_to_num(flux, nan=1.0, posinf=1.0, neginf=1.0)
    if rest:
        rest = [jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                for x in rest]

    # Flip transits upside-down if r[0] < 0
    flux = jnp.where(flip, 2.0 - flux, flux)

    return (flux, *rest)


@jax.jit
def harmonica_transit_quad_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                              u1=0., u2=0., r=jnp.array([0.1])):
    """ Harmonica transits with jax -- quadratic limb darkening.

    Parameters
    ----------
    times : ndarray
        1D array of model evaluation times [days].
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians].
    ecc : float, optional
        Eccentricity [], 0 <= ecc < 1. Default=0.
    omega : float, optional
        Argument of periastron [radians]. Default=0.
    u1, u2 : floats
        Quadratic limb-darkening coefficients.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. The length of r must be odd,
        and the final two coefficients must not both be zero.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].
    """
    return _harmonica_transit_common(
        jax_light_curve_quad_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r
    )[0]


@jax.jit
def _quad_ld_flux_and_derivatives(times, t0, period, a, inc, ecc, omega,
                                  u1, u2, r):
    """Return both the flux and its derivatives for the quadratic LD model.

    Intended for internal use and testing. This function bypasses the
    simplified public API and directly exposes both model outputs.

    Parameters
    ----------
    Same as `harmonica_transit_quad_ld`.

    Returns
    -------
    flux : ndarray
        Model flux values.
    d_flux_d_params : ndarray
        Derivatives of the flux with respect to all input parameters.
    """
    return _prepare_args_and_call_primitive(
        jax_light_curve_quad_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r
    )


def jax_light_curve_quad_ld_prim(times, *params):
    """ Define new JAX primitive. """
    return jax_light_curve_quad_ld_p.bind(times, *params)


def jax_light_curve_quad_ld_abstract_eval(abstract_times, *abstract_params):
    """ Define the abstract evaluation. """
    # Define first model output.
    abstract_model_eval = jax.core.ShapedArray(
        abstract_times.shape, abstract_times.dtype)

    # Define second model output.
    n_params = len(abstract_params)
    abstract_model_derivatives = jax.core.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype)

    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_quad_ld_xla_translation(ctx, timesc, *paramssc):
    """MLIR lowering for quadratic LD transit model."""

    # 1. Use full abstract value for input `times`
    timesc_aval = ctx.avals_in[0]
    data_type = timesc_aval.dtype
    shape = timesc_aval.shape

    # 2. Compute total number of time points (flattened)
    n_times = int(np.prod(shape))  # Ensure integer type
    n_times_const = ir_constant(n_times)

    # 3. Count r coefficients from parameters:
    #     total - 6 orbit - 2 LD
    n_rs = len(paramssc) - 6 - 2
    n_rs_const = ir_constant(n_rs)

    # 4. Output shapes:
    #    - 1D flux array of length n_times
    #    - 2D derivatives array of shape (n_times, n_params)
    output_shape_model_eval = ir.RankedTensorType.get(
        (n_times,), ir_dtype(data_type))
    shape_derivatives = (n_times, 6 + 2 + n_rs)
    output_shape_model_derivatives = ir.RankedTensorType.get(
        shape_derivatives, ir_dtype(data_type))

    return custom_call(
        b"jax_light_curve_quad_ld",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[n_times_const, n_rs_const, timesc, *paramssc],
        operand_layouts=[
            (), (), list(reversed(range(len(shape))))
        ] + [
            list(reversed(range(len(shape))))
        ] * len(paramssc),
        result_layouts=[
            [0],       # 1D flux output
            [1, 0]     # 2D df/dparams output
        ]
    ).results


def jax_light_curve_quad_ld_value_and_jvp(arg_values, arg_tangents):
    """ Evaluate the primal output and the tangents. """
    # Unpack parameter values and tangents.
    times, *args = arg_values
    _, *dargs = arg_tangents

    # Run the model to get the value and derivatives as designed.
    f, df_dz = jax_light_curve_quad_ld_prim(times, *args)

    # Compute grad.
    df = 0.
    for idx_pd, pd in enumerate(dargs):
        if isinstance(pd, ad.Zero):
            # This partial derivative is not required. It has been
            # set to a deterministic value.
            continue
        df += pd * df_dz[..., idx_pd]

    # Return valid zero tangent for the second output (Jacobian) as we are
    # not interested in using it for gradient-based inference.
    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


@jax.jit
def harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                   u1=0., u2=0., u3=0., u4=0.,
                                   r=jnp.array([0.1])):
    """ Harmonica transits with jax -- non-linear limb darkening.

    Parameters
    ----------
    times : ndarray
        1D array of model evaluation times [days].
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians].
    ecc : float, optional
        Eccentricity [], 0 <= ecc < 1. Default=0.
    omega : float, optional
        Argument of periastron [radians]. Default=0.
    u1, u2, u3, u4 : floats
        Non-linear limb-darkening coefficients.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. The length of r must be odd,
        and the final two coefficients must not both be zero.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].
    """
    return _harmonica_transit_common(
        jax_light_curve_nonlinear_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r
    )[0]


@jax.jit
def _nonlinear_ld_flux_and_derivatives(times, t0, period, a, inc, ecc, omega,
                                       u1, u2, u3, u4, r):
    """Return both the flux and its derivatives for the non-linear LD model.

    Intended for internal use and testing. This function bypasses the
    simplified public API and directly exposes both model outputs.

    Parameters
    ----------
    Same as `harmonica_transit_nonlinear_ld`.

    Returns
    -------
    flux : ndarray
        Model flux values.
    d_flux_d_params : ndarray
        Derivatives of the flux with respect to all input parameters.
    """
    f, df_dz = _prepare_args_and_call_primitive(
        jax_light_curve_nonlinear_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r
    )
    return f, df_dz


def jax_light_curve_nonlinear_ld_prim(times, *params):
    """ Define new JAX primitive. """
    return jax_light_curve_nonlinear_ld_p.bind(times, *params)


def jax_light_curve_nonlinear_ld_abstract_eval(abstract_times,
                                               *abstract_params):
    """ Define the abstract evaluation. """
    # Define first model output.
    abstract_model_eval = jax.core.ShapedArray(
        abstract_times.shape, abstract_times.dtype)

    # Define second model output.
    n_params = len(abstract_params)
    abstract_model_derivatives = jax.core.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype)

    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_nonlinear_xla_translation(ctx, timesc, *paramssc):
    """MLIR lowering for nonlinear LD transit model."""

    # 1. Use full abstract value for input `times`
    timesc_aval = ctx.avals_in[0]
    data_type = timesc_aval.dtype
    shape = timesc_aval.shape

    # 2. Compute total number of time points (flattened)
    n_times = int(np.prod(shape))  # Ensure it's Python int for ir_constant
    n_times_const = ir_constant(n_times)

    # 3. Count r coefficients from parameters:
    # total - 6 orbit - 4 limb darkening
    n_rs = len(paramssc) - 6 - 4
    n_rs_const = ir_constant(n_rs)

    # 4. Fix output shapes to match what the C++ backend returns:
    #    - A flat 1D flux array of length n_times
    #    - A 2D array of shape (n_times, n_params)
    output_shape_model_eval = ir.RankedTensorType.get(
        (n_times,), ir_dtype(data_type))
    shape_derivatives = (n_times, 6 + 4 + n_rs)
    output_shape_model_derivatives = ir.RankedTensorType.get(
        shape_derivatives, ir_dtype(data_type))

    return custom_call(
        b"jax_light_curve_nonlinear_ld",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[n_times_const, n_rs_const, timesc, *paramssc],
        operand_layouts=[
            (), (), list(reversed(range(len(shape))))
        ] + [
            list(reversed(range(len(shape))))
        ] * len(paramssc),
        result_layouts=[
            [0],       # 1D flux output
            [1, 0]     # 2D df/dparams output
        ]
    ).results


def jax_light_curve_nonlinear_ld_value_and_jvp(arg_values, arg_tangents):
    times, *args = arg_values
    _, *dargs = arg_tangents

    # Call the custom primitive while preventing JAX from trying to autodiff
    # through it. This is important because we're supplying explicit
    # derivatives (df_dz) below.
    f, df_dz = jax_light_curve_nonlinear_ld_prim(
        times, *args)

    # Sanitize backend outputs to avoid NaNs leaking into autodiff
    f = jnp.nan_to_num(f, nan=1.0, posinf=1.0, neginf=1.0)
    df_dz = jnp.nan_to_num(df_dz, nan=0.0, posinf=0.0, neginf=0.0)

    df = 0.
    for idx_pd, pd in enumerate(dargs):
        if isinstance(pd, ad.Zero):
            continue
        df += pd * df_dz[..., idx_pd]

    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


# Register the C++ models, bytes string required.
xla_client.register_custom_call_target(
    b'jax_light_curve_quad_ld',
    bindings.jax_registrations()['jax_light_curve_quad_ld']
)
xla_client.register_custom_call_target(
    b'jax_light_curve_nonlinear_ld',
    bindings.jax_registrations()['jax_light_curve_nonlinear_ld']
)


# Common utility function to call `_harmonica_transit_common`
# Handles parameter broadcasting, ripple validation, etc.
def _prepare_args_and_call_primitive(primitive_fn, times, param_list, r):
    return _harmonica_transit_common(primitive_fn, times, param_list, r)


# Create a primitive for quad ld.
jax_light_curve_quad_ld_p = Primitive('jax_light_curve_quad_ld')
jax_light_curve_quad_ld_p.multiple_results = True
jax_light_curve_quad_ld_p.def_abstract_eval(
    jax_light_curve_quad_ld_abstract_eval)
mlir.register_lowering(
    jax_light_curve_quad_ld_p, jax_light_curve_quad_ld_xla_translation,
    platform='cpu')
ad.primitive_jvps[jax_light_curve_quad_ld_p] = (
    jax_light_curve_quad_ld_value_and_jvp)

# Create a primitive for non-linear ld.
jax_light_curve_nonlinear_ld_p = Primitive(
    'jax_light_curve_nonlinear_ld')
jax_light_curve_nonlinear_ld_p.multiple_results = True
jax_light_curve_nonlinear_ld_p.def_abstract_eval(
    jax_light_curve_nonlinear_ld_abstract_eval)
mlir.register_lowering(
    jax_light_curve_nonlinear_ld_p, jax_light_curve_nonlinear_xla_translation,
    platform='cpu')
ad.primitive_jvps[jax_light_curve_nonlinear_ld_p] = (
    jax_light_curve_nonlinear_ld_value_and_jvp)
