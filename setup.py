from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='cubicsym',
    version='0.1',
    scripts=[p for p in map(str,Path("scripts/").rglob("*")) if Path(p).is_file() if "__pycache__" not in  str(p)],
    packages=['cubicsym'],
    url='https://github.com/Andre-lab/cubicsym',
    license='MIT',
    author='mads',
    author_email='mads.jeppesen@biochemistry.lu.se',
    description='Tools for using cubic symmetry with Rosetta',
    # FIXME: mpi could be an optional in the the future
	install_requires = [
            "mpi4py",
            "numpy",
            "pandas",
            "biopython",
            "pytest"
	]
)
