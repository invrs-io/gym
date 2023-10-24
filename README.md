# invrs-gym - A collection of inverse design challenges

## Overview
The `invrs_gym` package is an open-source gym containing a diverse set of photonic design challenges, which are relevant for a wide range of applications such as AR/VR, optical networking, LIDAR, and others.

Each of the challenges consists of a high-dimensional problem in which a physical structure (the photonic device) is optimized. The structure includes typically >10,000 degrees of freedom (DoF), generally including one or more arrays representing the structure or patterning of a layer, and may also include scalar variables representing e.g. layer thickness. In general, the DoF must satisfy certain constraints to be physical: thicknesses must be positive, and layer patterns must be _manufacturable_---they must not include features that are too small, or too closely spaced. 

In general, we seek optimization techniques that _reliably_ produce manufacturable, high-quality solutions and require reasonable compute resources.



## Example




Traditionally, a designer faced with such challenges would use their knowledge to define a low-dimensional solution space, and use gradient-free methods such as particle swarms to find a local optimum.


## Install
```
pip install invrs_gym
```

## Testing
Some tests are marked as slow and are skipped by default. To run these manually, use
```
pytest --runslow
```
