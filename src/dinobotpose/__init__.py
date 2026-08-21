"""DINObotPose — monocular robot pose and joint-angle estimation by iterative model fitting.

The modules here import each other by bare name (`from model_v4 import ...`), which the entry
points under scripts/ and train/ arrange by putting this directory on `sys.path`. Importing the
package does the same, so `import dinobotpose.solve` and the flat style both work and resolve to
the same modules either way.
"""
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

__version__ = "1.0.0"
