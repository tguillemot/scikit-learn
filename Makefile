# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(NOSETESTS) -s sklearn
test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-fixtures=_fixture doc/modules/

test-coverage:
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=sklearn sklearn

test: test-code test-doc

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

cython:
	find sklearn -name "*.pyx" | xargs $(CYTHON)

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *
