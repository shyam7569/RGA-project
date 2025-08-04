# RGA Optimization Code README

This README provides instructions for setting up and running the RGA (Real-coded Genetic Algorithm) optimization code, including the machine movement time optimization and minimization variants. The code is implemented in Python and uses the `pymoo` library for optimization tasks.

## Prerequisites

- **Python**: Ensure Python is installed on your system.
- **pymoo Library**: Install the `pymoo` library by following the steps below.
- **Operating System**: Instructions are provided for Windows; adapt for other OS as needed.

## Installation

1. **Open Command Prompt**:
   - Press `Windows + R`, type `cmd`, and press Enter.
   - Right-click the Command Prompt window and select "Run as Administrator".

2. **Install pymoo**:
   ```bash
   pip install pymoo
   ```
   Replace `sindh` with your system username if needed.

## Files Overview

- **task_24_June_2025.py**: Main RGA optimization code.
- **MHMtask_24_June_2025.py**: Machine movement time optimization code.
- **MHM2_24_June_2025.py**: Machine movement minimization code with input file.
- **user_def.txt**: Configuration file for specifying parameters and objective function paths.
- **obj_func.py**: File to define custom objective functions.
- **inputs.txt**: Input file for machine movement minimization parameters.

## Code Workflow

### task_24_June_2025.py
1. **Imports** (Lines 1-19): Imports necessary libraries and modules.
2. **Variable and Constraint Limits** (Lines 24-25): Modify maximum variables (default: 15) and constraints (default: 10) as needed.
3. **Input Validation** (Lines 27-91): Validates user inputs.
4. **Load Data from File** (Lines 94-132): Reads input data from a file.
5. **Interactive Input Collection** (Lines 134-251): Collects inputs interactively from the user.
6. **Text File Input Collection** (Lines 253-400): Reads inputs from a text file.
7. **Print Input File for Verification** (Lines 401-436): Displays input data for verification.
8. **Unpack Input Data** (Lines 438-467): Processes input data for optimization.
9. **Sampling** (Lines 469-491): Generates initial population using Latin Hypercube Sampling.
10. **Survival** (Lines 493-504): Implements survival selection.
11. **Tournament Selection** (Lines 505-518): Performs tournament-based selection.
12. **Algorithm Components** (Lines 519-522): Configures the genetic algorithm components.
13. **Custom Mutation** (Lines 524-551): Defines custom mutation operator.
14. **Custom Problem Definition** (Lines 553-585): Defines the optimization problem.
15. **Algorithm Execution** (Lines 587-610): Runs the optimization algorithm.
16. **Display Results** (Lines 612-645): Outputs the optimization results.
17. **Plotting** (Lines 646-668): Generates plots for visualization.

### MHMtask_24_June_2025.py
- **Purpose**: Optimizes machine movement time.
- **Parameter Customization**: Modify parameters in lines 439-455 as needed.
- **Save Changes**: Save the file after modifications (`Ctrl + S`).

### MHM2_24_June_2025.py
- **Purpose**: Machine movement minimization using an input file.
- **Input File Path**: Specify the path to `inputs.txt` in line 20.
- **Save Changes**: Save the file after modifications.

## Running the Code

1. **Navigate to the File Directory**:
   - Open Command Prompt.
   - Use the `cd` command to navigate to the directory containing the code files. For example:
     ```bash
     cd C:\Users\sindh\OneDrive\Desktop\New folder
     ```
     Replace `sindh` with your username and adjust the path as needed.

2. **Run the Code**:
   - For `task_24_June_2025.py`:
     ```bash
     python task_24_June_2025.py
     ```
   - For `MHMtask_24_June_2025.py`:
     ```bash
     python MHMtask_24_June_2025.py
     ```
   - For `MHM2_24_June_2025.py`:
     ```bash
     python MHM2_24_June_2025.py
     ```

3. **Input Selection**:
   - Choose between file-based or interactive input when prompted.
   - For file-based input, provide the full path to the input file (e.g., `inputs.txt`).
   - For interactive input, enter values as prompted.

## Configuration

### user_def.txt
- Specify the path to `obj_func.py` if using a custom objective function.
- Define parameters such as bounds and values.
- For standard g-functions, select `g` and omit the `obj_func.py` path.

### obj_func.py
- Define the objective function in the `obj` variable.
- Do not modify other parts of the file.

### inputs.txt
- Create or modify this file to include desired parameters for `MHM2_24_June_2025.py`.

## Notes
- Ensure all file paths are correct and accessible.
- For custom objective functions, update `obj_func.py` without altering its structure.
- Modify variable and constraint limits in `task_24_June_2025.py` (lines 24-25) if needed.
- The code supports both interactive and file-based input methods for flexibility.

## Troubleshooting
- If `pymoo` installation fails, ensure you have administrative privileges and a stable internet connection.
- Verify file paths in `user_def.txt` and `MHM2_24_June_2025.py` to avoid file-not-found errors.
- Check Python version compatibility with `pymoo` if errors occur.
