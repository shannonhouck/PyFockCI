from distutils.core import setup

setup(
    name='sf-ip-ea',
    version='0.1.0',
    author='Shannon E. Houck',
    author_email='shouck@vt.edu',
    packages=['sf-ip-ea', 'sf-ip-ea.test'],
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='Stand-alone RAS-SF-IP/EA code.',
    long_description=open('README.md').read(),
    install_requires=[
    ],
)
