def returnTrue(num):
    try:
        return True
    except:
        return False


def test_test():
    """This is a test of the test framework.  It should always pass.
    To make your own tests, copy this function, change the name, and add your own assertions.
    """
    assert returnTrue(6) == True