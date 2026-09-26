import os
import importlib.util

def resource_filename(package_or_requirement, resource_name):
    """
    A minimal polyfill for pkg_resources.resource_filename.
    Modern Python (3.12+) and newer setuptools (70+) have deprecated and removed pkg_resources.
    This function uses importlib to find the package directory and resolve the resource path.
    """
    spec = importlib.util.find_spec(package_or_requirement)
    if spec is None or spec.origin is None:
        raise ImportError(f"Package {package_or_requirement} not found")
    
    package_dir = os.path.dirname(spec.origin)
    return os.path.join(package_dir, resource_name)
