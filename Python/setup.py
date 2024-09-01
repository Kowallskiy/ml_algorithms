from setuptools import setup, Extension

setup(
    name='my_project',
    version='1.0.0',
    description='My custom machine learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[Extension('my_project', ['my_project.pyd'])],
    packages=['my_project'],
)
