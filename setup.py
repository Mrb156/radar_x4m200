from setuptools import setup, find_packages

setup(
    name='Processing of respiratory signals detected by a radar sensor',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/Mrb156/radar_x4m200',
    license='MIT',
    author='Barna Morvai',
    author_email='morvaibarna@gmail.com',
    description='This repository contains the source code for my thesis of signal processing from a radar system.',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        ]
)