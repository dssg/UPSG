from setuptools import setup, find_packages

setup(
        name='upsg',
        version='0.0.1',
        url='https://github.com/dssg/UPSG',
        author='Data Science For Social Good',
        description=('A set of tools and conventions to help data scientists '
                     'share code'),
        packages=find_packages(),
        install_requires=('numpy',
                          'scikit-learn', 
                          'matplotlib', 
                          'graphviz',
                          'SQLAlchemy',
                          'tables'),
        package_data={'upsg': ['bin/*.py', 'resources/*.json']},
        zip_safe=False)

