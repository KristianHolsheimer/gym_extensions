#!/usr/local/bin/blue-python2.7
import os
import setuptools


pwd = os.path.dirname(__file__)

install_requires = []
with open(os.path.join(pwd, 'requirements.txt')) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            install_requires.append(line)

with open(os.path.join(pwd, 'README.md')) as f:
    long_description = f.read().strip()

with open(os.path.join(pwd, 'version.txt')) as f:
    version = f.read().strip()

# main setup kw args
setup_kwargs = {
    'name': "gym-extensions",
    'version': version,
    'description': "Extensions to OpenAI gym",
    'long_description': long_description,
    'author': 'Kristian Holsheimer',
    'author_email': 'kristian.holsheimer@gmail.com',
    'license': 'BSD',
    'install_requires': install_requires,
    'classifiers': [
        'Development Status :: 1 - Planning',           # v0.1 - skeleton
        # 'Development Status :: 2 - Pre-Alpha',          # v0.2 - some basic functionality
        # 'Development Status :: 3 - Alpha',              # v0.3 - most functionality
        # 'Development Status :: 4 - Beta',               # v0.4 - most functionality + doc
        # 'Development Status :: 5 - Production/Stable',  # v1.0 - most functionality + doc + test
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        'Environment :: Other Environment',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    'zip_safe': True,
    'packages': setuptools.find_packages(),
}


if __name__ == '__main__':
    setuptools.setup(**setup_kwargs)
