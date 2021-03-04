from setuptools import setup


def long_description():
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md')) as readme_f:
        contents = readme_f.read()
        return contents


setup(
    name='notanorm',
    version='1.1.2',
    description='DB wrapper library',
    packages=['notanorm'],
    url="https://github.com/AtakamaLLC/notanorm",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    setup_requires=['wheel'],
    install_requires=[
        "mysqlclient",
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
        'Topic :: Utilities',
        ],
)
