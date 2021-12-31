from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    
setup(
    name='ablang',
    version='0.1.1',
    description='AbLang: A language model for antibodies.',
    license='BSD 3-clause license',
    maintainer='Tobias Hegelund Olsen',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer_email='tobias.olsen@stats.ox.ac.uk',
    include_package_data=True,
    packages=find_packages(include=('ablang', 'ablang.*')),
    install_requires=[
        'numpy',
        'requests',
        'fairseq',
        'torch>=1.6',
    ],
)