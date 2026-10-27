from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    
setup(
    name='ablang',
    version='0.2.4',
    description='AbLang: A language model for antibodies.',
    license='BSD 3-clause license',
    maintainer='Tobias Hegelund Olsen',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer_email='tobias.olsen@stats.ox.ac.uk',
    include_package_data=True,
    packages=find_packages(include=('ablang', 'ablang.*')),
    # ANARCI is needed at runtime for `align=True`, but it is deliberately
    # absent here.  The `anarci` package on PyPI is not an official release,
    # so ANARCI must come from Bioconda (`bioconda::anarci`).
    install_requires=[
        'requests',
        'torch>=1.6',
        'numpy',
        'pandas',
        'numba',
    ],
    entry_points={
        'console_scripts': ['ablang=ablang.cli:main'],
    },
)