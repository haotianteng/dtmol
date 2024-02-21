from setuptools import setup,find_packages,Extension
from setuptools.command.install import install
import numpy as np
import os
# read the contents of your README file
with open('README.md') as f:
    long_description = f.read()
print(long_description)
class CustomInstallCommand(install):
    def run(self):
        print("\nThis package is licensed under the GNU General Public License v3.0 (GPLv3).")
        print("Please refer to the LICENSE file for more information.\n")
        install.run(self)

install_requires=[
]
exec(open('dtmol/_version.py').read()) #readount the __version__ variable
setup(
  name = 'dtmol',
  packages = find_packages(exclude=["*.test", "*test.*", "test.*", "test"]),
  version = __version__,
  include_package_data=True,
  description = 'A probabilistic diffusion model for drug discovery.',
  author = 'Haotian Teng',
  author_email = 'havens.teng@gmail.com',
  url = 'https://github.com/haotianteng/dtMol', 
  download_url = 'https://github.com/haotianteng/dtMol/v0.0.1.tar.gz', 
  keywords = ['drug discovery', 'diffusion model', 'transformer'], 
  license="GPL 3.0",
  classifiers = ['License :: OSI Approved :: GNU General Public License v3 (GPLv3)'],
  install_requires=install_requires,
  entry_points={'console_scripts':['dtmol=dtMol.entry:main'],},
  long_description=long_description,
  include_dirs = [np.get_include()],
  long_description_content_type='text/markdown',
)
