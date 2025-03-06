# Changelog

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
