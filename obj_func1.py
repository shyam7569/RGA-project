import numpy as np

def user_objective_function(x1=None, x2=None, x3=None, x4=None, x5=None,
                           x6=None, x7=None, x8=None, x9=None, x10=None,
                           x11=None, x12=None, x13=None, x14=None, x15=None,
                           np=None, math=None, sin=None, cos=None, tan=None,
                           exp=None, log=None, sqrt=None, pi=None):
    """
    User-defined objective function using named variables.

    Parameters:
        x1, x2, ..., x15: float, optional input variables
        np, math, ..., pi: mathematical functions and constants
    
    Returns:
        float: The result of the objective function
    """
    # Example objective: x1^2 + x2^3
    obj = x1**2 + x2**3
    return obj