from distutils.core import setup

setup(
    name='notanorm',
    version='0.0.1',
    description='DB wrapper library',
    packages=['notanorm'],
    install_requires=[
        "mysqlclient",
    ]
)
