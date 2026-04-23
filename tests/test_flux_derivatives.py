import unittest
import numpy as np
import jax.numpy as jnp
from jax import jit, jvp, grad
import inspect

from harmonica.jax import (
    harmonica_transit_quad_ld,
    harmonica_transit_nonlinear_ld,
)
from harmonica.jax.custom_primitives import (
    _quad_ld_flux_and_derivatives, _nonlinear_ld_flux_and_derivatives
)


def get_param_index_by_name(func, name):
    """Get the positional index of a parameter in a function signature."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    return param_names.index(name)


def build_param_list(p):
    return [p['t0'], p['period'], p['a'], p['inc'], p['ecc'], p['omega'],
            *p['us'], p['rs']]


class TestFlux(unittest.TestCase):
    """ Test flux computations. """

    def __init__(self, *args, **kwargs):
        super(TestFlux, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(3)

        # Differential element and gradient error tolerance.
        self.epsilon = 1.e-8
        self.grad_tol = 1.e-4

        # Example params.
        self.t0 = 5.
        self.period = 10.
        self.a = 10.
        self.inc = 89. * np.pi / 180.
        self.ecc_zero = 0.
        self.ecc_non_zero = 0.1
        self.omega = 0.1 * np.pi / 180.

        # Input data structures.
        self.times = None
        self.fs = None

    def _build_test_data_structures(self, n_dp=100, start=2.5, stop=7.5):
        """ Build test input data structures. """
        self.times = np.ascontiguousarray(
            np.linspace(start, stop, n_dp), dtype=np.float64)
        self.fs = np.empty(self.times.shape, dtype=np.float64)

    def test_custom_jax_primitive_quad_ld(self):
        """ Test custom jax primitive for quadratic limb-darkening. """
        n_dp = 1000
        self._build_test_data_structures(n_dp=n_dp)
        u1 = 0.1
        u2 = 0.5
        r = jnp.array([0.1, -0.003, 0.])
        args = [self.t0, self.period, self.a, self.inc,
                self.ecc_non_zero, self.omega, u1, u2]

        times, *broadcasted_args = jnp.broadcast_arrays(self.times, *args)

        f, df_dz = jit(lambda t, *params:
                       _quad_ld_flux_and_derivatives(t, *params, r)
                       )(times, *broadcasted_args)
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(df_dz.ndim, 2)
        self.assertEqual(df_dz.shape[0], self.times.shape[0])
        self.assertEqual(df_dz.shape[1], 6 + 2 + 3)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertEqual(np.sum(np.isfinite(df_dz)), n_dp * (6 + 2 + 3))

        # Check JVP.
        # Build full argument values and tangents.
        arg_values = (self.times, *broadcasted_args, r)
        arg_tangents = (jnp.zeros_like(self.times),) + tuple(
            jnp.ones(n_dp) for _ in range(len(broadcasted_args))
            ) + (jnp.zeros_like(r),)
        # Define and call JVP
        der_jit = jit(lambda values, tangents: jvp(
            lambda t, *p: _quad_ld_flux_and_derivatives(t, *p),
            values, tangents))
        (f, df_dz), (jacobian_vp, _) = der_jit(arg_values, arg_tangents)
        self.assertEqual(f.shape, jacobian_vp.shape)
        self.assertEqual(np.sum(np.isfinite(jacobian_vp)), n_dp)

    def test_custom_jax_primitive_nonlinear_ld(self):
        """ Test custom jax primitive for non-linear limb-darkening. """
        n_dp = 1000
        self._build_test_data_structures(n_dp=n_dp)
        u1 = 0.33
        u2 = 0.96
        u3 = -0.68
        u4 = 0.17
        r = jnp.array([0.1, -0.003, 0.])
        args = [self.t0, self.period, self.a, self.inc,
                self.ecc_non_zero, self.omega, u1, u2, u3, u4]

        times, *broadcasted_args = jnp.broadcast_arrays(self.times, *args)

        f, df_dz = jit(lambda t, *params:
                       _nonlinear_ld_flux_and_derivatives(t, *params, r)
                       )(times, *broadcasted_args)
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(df_dz.ndim, 2)
        self.assertEqual(df_dz.shape[0], self.times.shape[0])
        self.assertEqual(df_dz.shape[1], 6 + 4 + 3)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertEqual(np.sum(np.isfinite(df_dz)), n_dp * (6 + 4 + 3))

        # Check JVP.
        # Build full argument values and tangents.
        arg_values = (self.times, *broadcasted_args, r)
        arg_tangents = (jnp.zeros_like(self.times),) + tuple(
            jnp.ones(n_dp) for _ in range(len(broadcasted_args))
            ) + (jnp.zeros_like(r),)
        # Define and call JVP
        der_jit = jit(lambda values, tangents: jvp(
            lambda t, *p: _nonlinear_ld_flux_and_derivatives(t, *p),
            values, tangents))
        (f, df_dz), (jacobian_vp, _) = der_jit(arg_values, arg_tangents)
        self.assertEqual(f.shape, jacobian_vp.shape)
        self.assertEqual(np.sum(np.isfinite(jacobian_vp)), n_dp)

    def test_api_jax_quad_ld(self):
        """ Test jax api for quadratic limb-darkening. """
        n_dp = 1000
        self._build_test_data_structures(n_dp=n_dp)

        # Check circle.
        f = harmonica_transit_quad_ld(
            self.times, self.t0, self.period, self.a, self.inc)
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertTrue(np.all(f >= 0.0))
        self.assertTrue(np.all(f <= 1.0))

        # Check n_rs = 3.
        f = harmonica_transit_quad_ld(
            self.times, self.t0, self.period, self.a, self.inc,
            self.ecc_zero, self.omega, u1=0.1, u2=0.5,
            r=jnp.array([0.1, -0.003, 0.]))
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertTrue(np.all(f >= 0.0))
        self.assertTrue(np.all(f <= 1.0))

        # Check n_rs = 7.
        f = harmonica_transit_quad_ld(
            self.times, self.t0, self.period, self.a, self.inc,
            self.ecc_zero, self.omega, u1=0.1, u2=0.5,
            r=jnp.array([0.1, -0.003, 0., 0., 0., 0., 0.001]))
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertTrue(np.all(f >= 0.0))
        self.assertTrue(np.all(f <= 1.0))

    def test_api_jax_nonlinear_ld(self):
        """ Test jax api for non-linear limb-darkening. """
        n_dp = 1000
        self._build_test_data_structures(n_dp=n_dp)

        # Create a jit-wrapped version of the function for testing
        f_jit = jit(harmonica_transit_nonlinear_ld)

        # Check circle.
        f = f_jit(
            self.times, self.t0, self.period, self.a, self.inc)
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertTrue(np.all(f >= 0.0))
        self.assertTrue(np.all(f <= 1.0))

        # Check n_rs = 3.
        f = f_jit(
            self.times, self.t0, self.period, self.a, self.inc,
            self.ecc_zero, self.omega, u1=0.1, u2=0.5, u3=-0.1, u4=0.,
            r=jnp.array([0.1, -0.003, 0.]))
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertTrue(np.all(f >= 0.0))
        self.assertTrue(np.all(f <= 1.0))

        # Check n_rs = 7.
        f = f_jit(
            self.times, self.t0, self.period, self.a, self.inc,
            self.ecc_zero, self.omega, u1=0.1, u2=0.5, u3=-0.1, u4=0.,
            r=jnp.array([0.1, -0.003, 0., 0., 0., 0., 0.001]))
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertTrue(np.all(f >= 0.0))
        self.assertTrue(np.all(f <= 1.0))

    def test_flux_derivative_quad_ld(self):
        """ Test flux derivative for quadratic limb-darkening. """
        param_names = ['t0', 'period', 'a', 'inc', 'ecc', 'omega', 'us', 'rs']
        for param_name in param_names:
            # Randomly generate trial light curves.
            for i in range(10):

                # Binomial probability of circular orbit.
                circular_bool = np.random.binomial(1, 0.5)
                if not circular_bool or param_name == 'ecc':
                    ecc = np.random.uniform(0., 0.6)
                else:
                    ecc = 0.

                # Uniform distributions of limb-darkening coeffs.
                u1 = np.random.uniform(0., 1.)
                u2 = np.random.uniform(-1, 1.)
                us = np.array([u1, u2])

                # Uniform distributions of transmission string coeffs.
                n_rs = 2 * (np.random.randint(3, 9) // 2) + 1
                a0 = np.random.uniform(0.05, 1.5)
                rs = np.append([a0], a0 * np.random.uniform(
                    -0.01, 0.01, n_rs - 1))
                rs = rs.astype(np.float64).flatten()  # ensure 1D array

                # Build parameter set.
                params = {'t0': np.random.uniform(2., 8.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.,
                          'ecc': ecc,
                          'omega': np.random.uniform(0., 2. * np.pi),
                          'us': us,
                          'rs': rs}

                # Compute fluxes.
                self._build_test_data_structures(n_dp=100)
                fs_a = harmonica_transit_quad_ld(
                    self.times, params['t0'], params['period'], params['a'],
                    params['inc'], params['ecc'], params['omega'],
                    *params['us'], params['rs'])

                # Update one parameter by epsilon.
                updated_params = params.copy()
                if param_name == 'us':
                    u_idx = np.random.randint(0, 2)
                    us_perturbed = us.copy()
                    us_perturbed[u_idx] += self.epsilon
                    updated_params['us'] = us_perturbed
                    _param_idx = get_param_index_by_name(
                        harmonica_transit_quad_ld, 'u1') + u_idx
                elif param_name == 'rs':
                    r_idx = np.random.randint(0, n_rs)
                    rs_perturbed = rs.copy()
                    rs_perturbed[r_idx] += self.epsilon
                    updated_params['rs'] = rs_perturbed
                    _param_idx = get_param_index_by_name(
                        harmonica_transit_quad_ld, 'r')
                else:
                    updated_params[param_name] = (
                        updated_params[param_name] + self.epsilon)
                    _param_idx = get_param_index_by_name(
                        harmonica_transit_quad_ld, param_name)

                # Get gradients.
                algebraic_gradients = []
                for j in range(self.times.shape[0]):
                    param_lst = build_param_list(params)
                    # Scalar-valued function used for autodiff via JAX
                    scalar_prim = lambda t, *p: \
                        harmonica_transit_quad_ld(t, *p)[0]
                    algebraic_gradients.append(
                        grad(scalar_prim, argnums=_param_idx)(
                            self.times[j], *param_lst))

                # Compute fluxes with updated parameter set.
                self._build_test_data_structures(n_dp=100)
                fs_b = harmonica_transit_quad_ld(
                    self.times, updated_params['t0'],
                    updated_params['period'], updated_params['a'],
                    updated_params['inc'], updated_params['ecc'],
                    updated_params['omega'], *updated_params['us'],
                    updated_params['rs'])

                # Check algebraic gradients match finite difference.
                res_iter = zip(fs_a, fs_b, algebraic_gradients)
                for res_idx, (f_a, f_b, algebraic_grad) in enumerate(res_iter):
                    finite_diff_grad = (f_b - f_a) / self.epsilon
                    if param_name == 'rs':
                        grad_component = algebraic_grad[r_idx]
                        grad_err = np.abs(finite_diff_grad - grad_component)
                    else:
                        grad_err = np.abs(finite_diff_grad - algebraic_grad)

                    self.assertLess(
                        grad_err, self.grad_tol,
                        msg='df/d{} failed lc {} dp {}.'.format(
                            param_name, i, res_idx))

    def test_flux_derivative_nonlinear_ld(self):
        """ Test flux derivative for non-linear limb-darkening. """
        param_names = ['t0', 'period', 'a', 'inc', 'ecc', 'omega', 'us', 'rs']
        for param_name in param_names:

            # Randomly generate trial light curves.
            for i in range(10):

                # Binomial probability of circular orbit.
                circular_bool = np.random.binomial(1, 0.5)
                if not circular_bool or param_name == 'ecc':
                    ecc = np.random.uniform(0., 0.9)
                else:
                    ecc = 0.

                # Uniform distributions of limb-darkening coeffs.
                us = np.random.uniform(-1., 1., 4)

                # Uniform distributions of transmission string coeffs.
                n_rs = 2 * (np.random.randint(3, 9) // 2) + 1
                a0 = np.random.uniform(0.05, 1.5)
                rs = np.append([a0], a0 * np.random.uniform(
                    -0.01, 0.01, n_rs - 1))
                rs = rs.astype(np.float64).flatten()  # ensure 1D array

                # Build parameter set.
                params = {'t0': np.random.uniform(2., 8.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.,
                          'ecc': ecc,
                          'omega': np.random.uniform(0., 2. * np.pi),
                          'us': us,
                          'rs': rs}

                # Compute fluxes.
                self._build_test_data_structures(n_dp=100)
                fs_a = harmonica_transit_nonlinear_ld(
                    self.times, params['t0'], params['period'], params['a'],
                    params['inc'], params['ecc'], params['omega'],
                    *params['us'], params['rs'])

                # Update one parameter by epsilon.
                updated_params = params.copy()
                if param_name == 'us':
                    u_idx = np.random.randint(0, 4)
                    us_perturbed = us.copy()
                    us_perturbed[u_idx] += self.epsilon
                    updated_params['us'] = us_perturbed
                    _param_idx = get_param_index_by_name(
                        harmonica_transit_nonlinear_ld, 'u1') + u_idx
                elif param_name == 'rs':
                    r_idx = np.random.randint(0, n_rs)
                    rs_perturbed = rs.copy()
                    rs_perturbed[r_idx] += self.epsilon
                    updated_params['rs'] = rs_perturbed
                    _param_idx = get_param_index_by_name(
                        harmonica_transit_nonlinear_ld, 'r')
                else:
                    updated_params[param_name] = (
                        updated_params[param_name] + self.epsilon)
                    _param_idx = get_param_index_by_name(
                        harmonica_transit_nonlinear_ld, param_name)

                # Get gradients.
                algebraic_gradients = []
                for j in range(self.times.shape[0]):
                    param_lst = build_param_list(params)
                    # Scalar-valued function used for autodiff via JAX
                    scalar_prim = lambda t, *p: \
                        harmonica_transit_nonlinear_ld(t, *p)[0]
                    algebraic_gradients.append(
                        grad(scalar_prim, argnums=_param_idx)(
                            self.times[j], *param_lst))

                # Compute fluxes with updated parameter set.
                self._build_test_data_structures(n_dp=100)
                fs_b = harmonica_transit_nonlinear_ld(
                    self.times, updated_params['t0'],
                    updated_params['period'], updated_params['a'],
                    updated_params['inc'], updated_params['ecc'],
                    updated_params['omega'], *updated_params['us'],
                    updated_params['rs'])

                # Check algebraic gradients match finite difference.
                res_iter = zip(fs_a, fs_b, algebraic_gradients)
                for res_idx, (f_a, f_b, algebraic_grad) in enumerate(res_iter):
                    finite_diff_grad = (f_b - f_a) / self.epsilon
                    if param_name == 'rs':
                        grad_component = algebraic_grad[r_idx]
                        grad_err = np.abs(finite_diff_grad - grad_component)
                    else:
                        grad_err = np.abs(finite_diff_grad - algebraic_grad)

                    self.assertLess(
                        grad_err, self.grad_tol,
                        msg='df/d{} failed lc {} dp {}.'.format(
                            param_name, i, res_idx))

    def test_nan_at_near_zero_ripples(self):
        """Ensure no NaNs when ripple coefficients are zero or near-zero."""
        self._build_test_data_structures(n_dp=100)
        r_zero = jnp.array([0.1, 0., 0.])
        r_near_zero = jnp.array([0.1, 1e-12, -1e-12])
        args = [self.t0, self.period, self.a, self.inc,
                self.ecc_zero, self.omega, 0.1, 0.5]

        # Check exactly zero ripples
        f, df_dz = _quad_ld_flux_and_derivatives(self.times, *args, r_zero)
        self.assertTrue(jnp.all(jnp.isfinite(f)))
        self.assertTrue(jnp.all(jnp.isfinite(df_dz)))

        # Check near-zero ripples
        f, df_dz = _quad_ld_flux_and_derivatives(self.times, *args,
                                                 r_near_zero)
        self.assertTrue(jnp.all(jnp.isfinite(f)))
        self.assertTrue(jnp.all(jnp.isfinite(df_dz)))

    def test_gradients_near_zero_coefficients(self):
        """Check gradient stability near zero ripple coefficients."""
        self._build_test_data_structures(n_dp=10)
        us = [0.1, 0.5]
        r = jnp.array([0.1, 1e-10, -1e-10])
        args = [self.t0, self.period, self.a, self.inc,
                self.ecc_zero, self.omega, *us, r]

        def scalar_flux_wrt_r1(t, *args):
            # Partial w.r.t. r[1], the first ripple coefficient
            r = args[-1].at[1].set(args[-1][1])  # just to clarify
            return _quad_ld_flux_and_derivatives(t, *args[:-1], r)[0].item()

        grad_r1 = grad(scalar_flux_wrt_r1, argnums=-1)
        for t in self.times:
            dfdx = grad_r1(t, *args)
            self.assertTrue(jnp.all(jnp.isfinite(dfdx)))


if __name__ == '__main__':
    unittest.main()
