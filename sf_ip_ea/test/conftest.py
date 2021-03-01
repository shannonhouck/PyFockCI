import pytest


def pytest_configure(config):
    # Register marks to avoid warnings in installed testing
    # sync with setup.cfg
    config.addinivalue_line("markers", "blochtest")
    config.addinivalue_line("markers", "davidsontest")
    config.addinivalue_line("markers", "dftest")
    config.addinivalue_line("markers", "frozentest")
    config.addinivalue_line("markers", "methodtest")


@pytest.fixture(scope="function", autouse=True)
def set_up():
    import psi4

    psi4.core.clean()
    # psi4.core.clean_timers()  # uncomment when v1.4 is lowest supported
    psi4.core.clean_options()
    psi4.set_output_file("pytest_output.dat", True)
