import setuptools

setuptools.setup(name='RandomForest',
      version="0.1git status.0",
      url = "https://github.com/stkobsar/RandomForest.git",
      description='Random Forest algorithm use case',
      author='Stephi Kobsar',
      author_email='stkobsar7@gmail.com',
      packages=setuptools.find_packages(),
      install_requires=["matplotlib", "scipy", "numpy", "seaborn", "sklearn"],

     )