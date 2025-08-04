# Import required libraries for numerical operations and mathematical functions
import numpy as np
import math

# Function to map input array elements to variable names for the objective function
def map_variables(x):
    """
    Maps elements of input array x to a dictionary with keys 'x1', 'x2', ..., 'x15'.
    
    Parameters:
        x (numpy.ndarray or list): Array of variable values, where x[i] corresponds to xi+1.
    
    Returns:
        dict: Dictionary mapping 'x1' to x[0], 'x2' to x[1], ..., up to 'x15'.
              If x has fewer elements than 15, remaining variables are set to 0.
    """
    # Create a dictionary using dictionary comprehension
    # For each index i (0 to 14), map 'xi+1' to x[i] if i < len(x), else 0
    return {f'x{i+1}': x[i] if i < len(x) else 0 for i in range(15)}

# User-defined objective function for optimization
def user_objective(x1=None, x2=None, x3=None, x4=None, x5=None,
                   x6=None, x7=None, x8=None, x9=None, x10=None,
                   x11=None, x12=None, x13=None, x14=None, x15=None):
    """
    Computes the objective function value based on input variables x1 and x2.
    This function is designed to be flexible, accepting up to 15 variables,
    but only uses x1 and x2 for the calculation.
    
    Parameters:
        x1, x2, ..., x15 (float or None): Input variables, default to None.
                                          Only x1 and x2 are used in the computation.
    
    Returns:
        float: Objective function value, calculated as x1^2 + x2^3.
    """
    # Compute the objective function: x1 squared plus x2 cubed
    obj = x1**2 - x2 - 6
    return obj

# Wrapper function to evaluate the objective function using an input array
def objective_function(x):
    """
    Evaluates the user-defined objective function by mapping input array x to variables.
    Serves as the interface between the optimization algorithm and user_objective.
    
    Parameters:
        x (numpy.ndarray or list): Array of variable values (e.g., [x1, x2, x3]).
    
    Returns:
        float: Objective function value computed by user_objective.
    
    Raises:
        ValueError: If user_objective returns a non-numeric value or if an error occurs
                    during evaluation.
    """
    # Map input array x to a dictionary of variables (e.g., {'x1': x[0], 'x2': x[1], ...})
    vars_dict = map_variables(x)
    try:
        # Call user_objective with mapped variables as keyword arguments
        result = user_objective(**vars_dict)
        # Validate that the result is a numeric scalar (int, float, or numpy number)
        if not isinstance(result, (int, float, np.number)):
            raise ValueError(f"Objective function returned non-numeric value: {type(result)}")
        return result
    except Exception as e:
        # Raise a ValueError with details if evaluation fails
        raise ValueError(f"Error evaluating user objective function: {str(e)}")