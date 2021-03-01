from distutils.core import setup
import setuptools


def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("sf_ip_ea")


setup(
    name='sf_ip_ea',
    version=version,
    cmdclass=cmdclass,
    author='Shannon E. Houck',
    author_email='shouck@vt.edu',
    packages=setuptools.find_packages(),
    include_package_data=True,
    scripts=[],
    url='',
    license='LICENSE.txt',
    install_requires=[
        "numpy>=1.7",
        "scipy",
    ],
    extras_require={
        "tests": ["pytest",],
    },
    tests_require=["pytest",],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    zip_safe=False,
    description='Stand-alone RAS-SF-IP/EA code.',
    long_description=open('README.md').read(),
)
