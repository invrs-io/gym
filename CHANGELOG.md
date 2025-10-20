# Changelog

## Unreleased

## 1.6.2
- Update readme

## 1.6.0
- Remove optimizations that make use of `jax.ensure_compile_time_eval`. Bugs introduced in newer versions of jax forced gym to be pinned to an earlier jax version. With the optimizations removed, compatibility with the latest jax versions is restored.
- Ensure that the `fixed_solid` or `fixed_void` attributes of density arrays are always numpy arrays. If these are jax arrays, jax can hang while compiling. The latest version of `totypes` also validates that the attributes are numpy arrays.
- Modify the extractor challenge, parallelizing the eigensolve to improve performance.
- Update for use of fmmax 1.4.0, which shifts the coordinates associated with arrays defined on the unit cell. This motivates commensurate shifts in sources for challenges making use of localized sources, and slight changes in regression test reference values.

## 1.5.1 (April 7, 2025)
- Library challenge: fix inadvertent conversion of metasurface thickness from scalar to 1D array.

## 1.5.0 (March 17, 2025)
- Update usage of fmmax for the 1.0.0 api.

## 1.4.11 (March 6, 2025)
- Remove keyword arguments in call to `fmmax.fields.time_average_z_poynting_flux` to accommodate an upcoming change to the fmmax api.

## 1.4.10 (March 3, 2025)
- FMMAX has added a function `fields.time_average_z_poynting_flux`, which makes some private functions defined in this package redundant. Remove the private functions and use the FMMAX function instead.

## 1.4.9 (January 24, 2025)
- Accommodate float-typed wavelength in the `utils.materials` module.

## 1.4.8 (January 17, 2025)
- Update use of `jax.pure_callback` to detect the jax version, and select the appropriate behavior under vmap.

## 1.4.7 (January 17, 2025)
- Update `utils.materials` to ensure no type promotion occurs when computing permittivity. For example, when wavelength has type `float32`, permittivity should have type `complex64` even if 64 bit precision is enabled.

## 1.4.6 (January 15, 2025)
- Update dependencies specified in `pyproject.toml` to allow jax versions equal to or greater than 0.4.27.
