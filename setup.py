from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Content based search engine for anime.'
LONG_DESCRIPTION = 'Library that creates search engine and searches through queries to get desired anime.'

setup(
        name="Yuno",
        version=VERSION,
        author="IAmParadox",
        author_email="yunogasai.search@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],

        keywords=['anime', 'Search Engine'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)

