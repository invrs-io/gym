# invrs-gym
<a href="https://invrs-io.github.io/gym/"><img src="https://img.shields.io/badge/Docs-blue.svg"/></a>
![Continuous integration](https://github.com/invrs-io/gym/actions/workflows/build-ci.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/invrs-gym)

## Overview
The `invrs_gym` package is an open-source gym containing a diverse set of photonic design challenges, which are relevant for a wide range of applications such as AR/VR, optical networking, LIDAR, and others. For a full description of the gym, see the [manuscript](https://arxiv.org/abs/2410.24132).

![invrs-gym challenge examples](https://github.com/invrs-io/gym/blob/main/docs/img/challenges.png?raw=true)

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
The current list of challenges is below.

- The **metagrating** challenge involves design of a large-angle beam deflector and is based on the [Metagrating3D](https://github.com/NanoComp/photonics-opt-testbed/tree/main/Metagrating3D) problem from "[Validation and characterization of algorithms and software for photonics inverse design](https://opg.optica.org/josab/abstract.cfm?uri=josab-41-2-A161)" by Chen et al.
- The **diffractive splitter** challenge involves design of a diffractive optic and is based on "[Design and Rigorous Analysis of Non-Paraxial Diffractive Beam Splitter](https://www.lighttrans.com/use-cases/application/design-and-rigorous-analysis-of-non-paraxial-diffractive-beam-splitter.html)", a LightTrans case study.
- The **meta-atom library** challenge is bassed on "[Dispersion-engineered metasurfaces reaching broadband 90% relative diffraction efficiency](https://www.nature.com/articles/s41467-023-38185-2)" by Chen et al., and involves the design of 8 meta-atoms for polarization-insensitive broadband large-area metasurfaces.
- The **bayer sorter** challenge involves the design of metasurface that replaces the color filter in an image sensor, and is based on "[Pixel-level Bayer-type colour router based on metasurfaces](https://www.nature.com/articles/s41467-022-31019-7)" by Zou et al.
- The **metalens** challenge involves design of a 1D achromatic metalens and is based on the [RGB Metalens](https://github.com/NanoComp/photonics-opt-testbed/tree/main/RGB_metalens) problem from "[Validation and characterization of algorithms and software for photonics inverse design](https://opg.optica.org/josab/abstract.cfm?uri=josab-41-2-A161)" by Chen et al.
- The **ceviche** challenges are jax-wrapped versions of the [Ceviche Challenges](https://github.com/google/ceviche-challenges) open-sourced by Google, with defaults matching "[Inverse Design of Photonic Devices with Strict Foundry Fabrication Constraints](https://pubs.acs.org/doi/10.1021/acsphotonics.2c00313)" by Schubert et al.
- The **photon extractor** challenge is based on "[Inverse-designed photon extractors for optically addressable defect qubits](https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-12-1805)" by Chakravarthi et al., and involves design of nanostructures to increase photon collection efficiency for quantum information processing applications.


## Install
```
pip install invrs_gym
```

## Citing the invrs-gym
If you use the gym for your research, please cite,

```
@misc{schubert2024invrsgymtoolkitnanophotonicinverse,
      title={invrs-gym: a toolkit for nanophotonic inverse design research},
      author={Martin F. Schubert},
      year={2024},
      eprint={2410.24132},
      archivePrefix={arXiv},
      primaryClass={physics.optics},
      url={https://arxiv.org/abs/2410.24132},
}
```

Please also cite the original paper in which the challenge used was introduced (click to expand).

<details>
<summary>Metagrating challenge</summary>

```
@article{chen2024validation,
  title={Validation and characterization of algorithms and software for photonics inverse design},
  author={Chen, Mo and Christiansen, Rasmus E and Fan, Jonathan A and I{\c{s}}iklar, G{\"o}ktu{\u{g}} and Jiang, Jiaqi and Johnson, Steven G and Ma, Wenchao and Miller, Owen D and Oskooi, Ardavan and Schubert, Martin F, and Wang, Fengwen and Williamson, Ian A D and Xue, Wenjin and Zou, You},
  journal={JOSA B},
  volume={41},
  number={2},
  pages={A161--A176},
  year={2024},
  publisher={Optica Publishing Group}
}
```

</details>
<details>
<summary>Diffractive splitter challenge</summary>

```
@misc{LightTrans,
  author = {LightTrans},
  title = {Design and Rigorous Analysis of Non-Paraxial Diffractive Beam Splitter},
  howpublished = {\url{https://www.lighttrans.com/use-cases/application/design-and-rigorous-analysis-of-non-paraxial-diffractive-beam-splitter.html}},
  note = {Version: 3.1},
}
```

</details>
<details>
<summary>Meta-atom library challenge</summary>

```
@article{chen2023dispersion,
  title={Dispersion-engineered metasurfaces reaching broadband 90\% relative diffraction efficiency},
  author={Chen, Wei Ting and Park, Joon-Suh and Marchioni, Justin and Millay, Sophia and Yousef, Kerolos MA and Capasso, Federico},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={2544},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

</details>
<details>
<summary>Bayer sorter challenge</summary>

```
@article{zou2022pixel,
  title={Pixel-level Bayer-type colour router based on metasurfaces},
  author={Zou, Xiujuan and Zhang, Youming and Lin, Ruoyu and Gong, Guangxing and Wang, Shuming and Zhu, Shining and Wang, Zhenlin},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={3288},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

</details>
<details>
<summary>Metalens challenge</summary>

```
@article{chen2024validation,
  title={Validation and characterization of algorithms and software for photonics inverse design},
  author={Chen, Mo and Christiansen, Rasmus E and Fan, Jonathan A and I{\c{s}}iklar, G{\"o}ktu{\u{g}} and Jiang, Jiaqi and Johnson, Steven G and Ma, Wenchao and Miller, Owen D and Oskooi, Ardavan and Schubert, Martin F, and Wang, Fengwen and Williamson, Ian A D and Xue, Wenjin and Zou, You},
  journal={JOSA B},
  volume={41},
  number={2},
  pages={A161--A176},
  year={2024},
  publisher={Optica Publishing Group}
}
```

</details>
<details>
<summary>Ceviche challenges</summary>

```
@article{chen2024validation,
  title={Validation and characterization of algorithms and software for photonics inverse design},
  author={Chen, Mo and Christiansen, Rasmus E and Fan, Jonathan A and I{\c{s}}iklar, G{\"o}ktu{\u{g}} and Jiang, Jiaqi and Johnson, Steven G and Ma, Wenchao and Miller, Owen D and Oskooi, Ardavan and Schubert, Martin F, and Wang, Fengwen and Williamson, Ian A D and Xue, Wenjin and Zou, You},
  journal={JOSA B},
  volume={41},
  number={2},
  pages={A161--A176},
  year={2024},
  publisher={Optica Publishing Group}
}
@article{schubert2022inverse,
  title={Inverse design of photonic devices with strict foundry fabrication constraints},
  author={Schubert, Martin F and Cheung, Alfred KC and Williamson, Ian AD and Spyra, Aleksandra and Alexander, David H},
  journal={ACS Photonics},
  volume={9},
  number={7},
  pages={2327--2336},
  year={2022},
  publisher={ACS Publications}
}
```

</details>

<details>
<summary>Photon extractor challenge</summary>

```
@article{chakravarthi2020inverse,
  title={Inverse-designed photon extractors for optically addressable defect qubits},
  author={Chakravarthi, Srivatsa and Chao, Pengning and Pederson, Christian and Molesky, Sean and Ivanov, Andrew and Hestroffer, Karine and Hatami, Fariba and Rodriguez, Alejandro W and Fu, Kai-Mei C},
  journal={Optica},
  volume={7},
  number={12},
  pages={1805--1811},
  year={2020},
  publisher={Optica Publishing Group}
}
```

</details>

## Known issues
- jax versions above 0.4.35 have a bug which causes challenges making use of `jax.ensure_compile_time_eval` to fail.


## Testing
Some tests are marked as slow and are skipped by default. To run these manually, use
```
pytest --runslow
```
