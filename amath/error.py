
class MathError(Exception):
    """
    Math error
    
    # Cause
    1. Result overflow
    2. Math equation is invalid
    """
    
class TimeOutError(Exception):
    """
    Calculate timeout
    """
    
class CalculateLimitExceededError(Exception):
    """
    Calculate limit exceeded
    """
    
class Terminate(Exception):
    """
    Program terminate
    """
class InvalidArgument(Exception):
    """
    Invalid argument input
    """

class CallSystemFunctionWarning(Warning):
    pass