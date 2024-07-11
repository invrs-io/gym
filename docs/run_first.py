# Some dependencies of the gym print the first time they are imported, e.g. the
# `refractiveindex` project. In the docs github workflow, running this script
# before any of the notebooks avoids these print statements showing up in the
# output of the notebooks.
from invrs_gym import challenges as challenges

if __name__ == "__main__":
    pass
