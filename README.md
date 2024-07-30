# invrs-gym
`v1.1.2`

## Overview
The `invrs_gym` package is an open-source gym containing a diverse set of photonic design challenges, which are relevant for a wide range of applications such as AR/VR, optical networking, LIDAR, and others.

Each of the challenges consists of a high-dimensional problem in which a physical structure (the photonic device) is optimized. The structure includes typically >10,000 degrees of freedom (DoF), generally including one or more arrays representing the structure or patterning of a layer, and may also include scalar variables representing e.g. layer thickness. In general, the DoF must satisfy certain constraints to be physical: thicknesses must be positive, and layer patterns must be _manufacturable_---they must not include features that are too small, or too closely spaced.

In general, we seek optimization techniques that _reliably_ produce manufacturable, high-quality solutions and require reasonable compute resources. Among the techniques that could be applied are topology optimization, inverse design, and AI-guided design.

`invrs_gym` is intended to facilitate research on such methods within the jax ecosystem. It includes several challenges that have been used in previous works, so that researchers may directly compare their results to those of the literature. While some challenges are test problems (e.g. where the structure is two-dimensional, which is unphysical but allows fast simulation), others are actual problems that are relevant e.g. for quantum computing or 3D sensing.

## Key concepts
The key types of the challenge are the `Challenge` and `Component` objects.

The `Component` represents the physical structure to be optimized, and has some intended excitation or operating condition (e.g. illumination with a particular wavelength from a particular direction). The `Component` includes methods to obtain initial parameters, and to compute the _response_ of a component to the excitation.

Each `Challenge` has a `Component` as an attribute, and also has a target that can be used to determine whether particular parameters "solve" the challenge. The `Challenge` also provides functions to compute a scalar loss for use with gradient-based optimization, and additional metrics.

## Example
```python
# Select the challenge.
challenge = invrs_gym.challenges.ceviche_lightweight_waveguide_bend()

# Define loss function, which also returns auxilliary quantities.
def loss_fn(params):
    response, aux = challenge.component.response(params)
    loss = challenge.loss(response)
    eval_metric = challenge.eval_metric(response)
    metrics = challenge.metrics(response, params, aux)
    return loss, (response, eval_metric, metrics, aux)

value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

# Select an optimizer.
opt = invrs_opt.density_lbfgsb(beta=4)

# Generate initial parameters, and use these to initialize the optimizer state.
params = challenge.component.init(jax.random.PRNGKey(0))
state = opt.init(params)

# Carry out the optimization.
for i in range(steps):
    params = opt.params(state)
    (value, (response, eval_metric, metrics, aux)), grad = value_and_grad_fn(params)
    state = opt.update(grad=grad, value=value, params=params, state=state)
```
With some plotting, this code will produce the following waveguide bend:

![Animated evolution of waveguide bend design](https://github.com/invrs-io/gym/blob/main/docs/img/waveguide_bend.gif?raw=true)

## Challenges
The current list of challenges is below. Check out the notebooks for ready-to-go examples of each.

- The **bayer sorter** chhallenge involves the design of metasurface that replaces the color filter in an image sensor, and is based on "[Pixel-level Bayer-type colour router based on metasurfaces](https://www.nature.com/articles/s41467-022-31019-7)" by Zou et al.
- The **diffractive splitter** challenge involves designing a non-paraxial diffractive beamsplitter useful for 3D sensing, as discussed in [LightTrans documentation](https://www.lighttrans.com/use-cases/application/design-and-rigorous-analysis-of-non-paraxial-diffractive-beam-splitter.html).
- The **ceviche** challenges are jax-wrapped versions of the [Ceviche Challenges](https://github.com/google/ceviche-challenges) open-sourced by Google, with defaults matching "[Inverse Design of Photonic Devices with Strict Foundry Fabrication Constraints](https://pubs.acs.org/doi/10.1021/acsphotonics.2c00313)" by Schubert et al. These were also studied by Ferber et al. in "[SurCo: Learning Linear SURrogates for COmbinatorial Nonlinear Optimization Problems](https://proceedings.mlr.press/v202/ferber23a/ferber23a.pdf)" by Ferber et al.
- The **meta-atom library** challenge is baed on "[Dispersion-engineered metasurfaces reaching broadband 90% relative diffraction efficiency](https://www.nature.com/articles/s41467-023-38185-2)" by Chen et al., and involves the design of 8 meta-atoms for
- The **metagrating** challenge is a re-implementation of the [Metagrating3D](https://github.com/NanoComp/photonics-opt-testbed/tree/main/Metagrating3D) problem using the [fmmax](https://github.com/facebookresearch/fmmax) simulator.
- The **metalens** challenge is a re-implemenation of the [RGB Metalens](https://github.com/NanoComp/photonics-opt-testbed/tree/main/RGB_metalens) problem using the [fmmax](https://github.com/facebookresearch/fmmax) simulator.
constructing a broadband, polarization-insensitive grating.
- The **photon extractor** challenge is based on "[Inverse-designed photon extractors for optically addressable defect qubits](https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-12-1805)" by Chakravarthi et al., and aims to create structures that increase photon extraction efficiency for quantum applications.


## Install
```
pip install invrs_gym
```

## Testing
Some tests are marked as slow and are skipped by default. To run these manually, use
```
pytest --runslow
```
