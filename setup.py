from setuptools import setup

setup(
    name='pyctl',
    version='0.2.1',
    description='Control with Python',
    url='https://github.com/mtguerreiro/pyctl.git',
    author='Marco Guerreiro',
    author_email='marcotulio.guerreiro@gmail.com',
    packages=['pyctl'],
    install_requires=[
      'numpy',
      'scipy',
      'matplotlib',
      'qpsolvers[quadprog]',
      'osqp==0.6.7.post3',
    ],
)
