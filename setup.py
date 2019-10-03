from setuptools import setup


def long_description():
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md')) as readme_f:
        contents = readme_f.read()
        return contents


setup(
    name='notanorm',
    version='0.0.4',
    description='DB wrapper library',
    packages=['notanorm'],
    long_description=long_description(),
    long_description_content_type="text/markdown",
    setup_requires=['wheel'],
    install_requires=[
        "mysqlclient",
    ]
)
