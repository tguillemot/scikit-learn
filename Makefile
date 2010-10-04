# simple makefile to simplify repetetive build env management tasks under posix

PYTHON ?= python
NOSETESTS ?= nosetests

all: clean inplace test

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean: clean-build clean-pyc clean-so

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test: in
	$(PYTHON) test.py
test-doc:
	$(NOSETESTS) --doctest-extension=rst doc/
