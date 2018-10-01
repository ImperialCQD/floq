import setuptools

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

packages = setuptools.find_packages(exclude=('tests', 'examples'))

setuptool.setup(
    name='floq',
    version='0.1.0',
    description='Floquet control code',
    long_description=readme,
    author='Imperial College London CQD Group',
    author_email='jake.lishman16@imperial.ac.uk',
    url='https://www.github.com/ImperialCQD/floq',
    license=license,
    packages=packages,
)
