from setuptools import setup,find_packages

with open('requirements.txt') as requirement_file:
    requirements = requirement_file.read().split()

setup(
name='ldgan',
description="Learning Degradation Using Generative Adversarial Network",
version="1.0.0",
author="Molefe",
package_dir = { 
"":"src"
}
)
