from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='cubicsym',
    version='0.1',
    scripts=[p for p in map(str,Path("scripts/").rglob("*")) if Path(p).is_file() if "__pycache__" not in  str(p) and "__init__.py" not in  str(p)] + ["scripts/make_symmdef_file.pl"],
    packages=find_packages(where=".", exclude=("tests",)),
    #packages=['cubicsym'],
    package_data={'cubicsym' :['data/I/*', 'data/O/*', 'data/T/*']},
    url='https://github.com/Andre-lab/cubicsym',
    license='MIT',
    author='mads',
    author_email='mads.jeppesen@biochemistry.lu.se',
    description='Tools for using cubic symmetry with Rosetta and EvoDOCK',
	install_requires = [
            "symmetryhandler @ git+https://github.com/Andre-lab/symmetryhandler@903b10a7580522ce118a16002096ea8bc03b63cb",
            "requests",
            "numpy",
            "pandas",
            "biopython",
            "pytest",
            "scipy",
            "pyyaml"
	],
)
