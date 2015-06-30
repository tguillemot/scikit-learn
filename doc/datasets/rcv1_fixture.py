"""Fixture module to skip the datasets loading when offline

The RCV1 data is rather large and some CI workers such as travis are
stateless hence will not cache the dataset as regular sklearn users would do.

The following will skip the execution of the rcv1.rst doctests
if the proper environment variable is configured (see the source code of
check_skip_network for more details).

"""
from sklearn.utils.testing import check_skip_network


def setup_module():
    check_skip_network()
