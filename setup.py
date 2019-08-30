from distutils.core import setup

setup(
    name='sf_ip_ea',
    version='0.1.0',
    author='Shannon E. Houck',
    author_email='shouck@vt.edu',
    packages=['sf_ip_ea', 'sf_ip_ea.test'],
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='Stand-alone RAS-SF-IP/EA code.',
    long_description=open('README.md').read(),
)
