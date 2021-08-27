from setuptools import find_packages, setup

setup(
    name='otdd',
    version='0.1.0',
    description='Gridworlds library',
    author='David Alvarez-Melis & Aldo Pacchiano',
    license='MIT',
    packages=find_packages(),
    install_requires=[
      'numpy',
      'scipy',
      'matplotlib',
      'tqdm',
      'pot',
      'requests',
      'torch',
      'torchvision',
      'torchtext',
      'attrdict',
      'opentsne',
      'seaborn',
      'scikit-learn',
      'pandas'
    ],
    include_package_data=True,
    zip_safe=False
)