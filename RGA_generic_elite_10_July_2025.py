import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.sampling import Sampling
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.problems import get_problem as pymoo_get_problem
import math
import ast
import os
import importlib.util
import sys
import datetime
from pymoo.core.survival import Survival

# ========== INPUT WITH VALIDATION ==========
def get_int_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = int(input(prompt))
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Please enter an integer between {min_val} and {max_val}.")
            else:
                return val
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

def snap_discrete_value(Vk, Vmin, Vmax, lc=None, disc_values=None):
    if lc is not None:
        Vk = max(min(Vk, Vmax), Vmin)
        NV = math.ceil((Vmax - Vmin) / lc)
        Q = (Vk - Vmin) / lc
        Q = math.floor(Q) + 1 if Q >= (math.floor(Q) + 0.5) else math.floor(Q)
        Vk = Vmin + lc * Q
        Vk = max(min(Vk, Vmax), Vmin)
    elif disc_values is not None:
        Vk = np.clip(Vk, Vmin, Vmax)
        Rk = (Vk - Vmin) / (Vmax - Vmin) if Vmax != Vmin else 0
        Vk_index = min(int(round(Rk * (len(disc_values) - 1))), len(disc_values) - 1)
        Vk = disc_values[Vk_index]
    return Vk

def load_objective_function(file_path, n_var):
    try:
        if not os.path.exists(file_path):
            print(f"Error: The file {file_path} does not exist.")
            sys.exit(1)
        spec = importlib.util.spec_from_file_location("module", file_path)
        obj_func = importlib.util.module_from_spec(spec)
        sys.modules["module"] = obj_func
        spec.loader.exec_module(obj_func)
        if not hasattr(obj_func, 'objective_function'):
            print(f"Error: {file_path} must define a function named 'objective_function'.")
            sys.exit(1)
        import inspect
        sig = inspect.signature(obj_func.objective_function)
        if len(sig.parameters) != 1:
            print(f"Error: objective_function must take exactly one parameter (x).")
            sys.exit(1)
        test_x = np.zeros(n_var)
        try:
            result = obj_func.objective_function(test_x)
            if not isinstance(result, (int, float, np.number)):
                print(f"Error: objective_function must return a scalar numeric value, got {type(result)}.")
                sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to evaluate objective_function with test input: {str(e)}")
            sys.exit(1)
        return obj_func.objective_function
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        sys.exit(1)

def validate_expression(expr, n_var, allow_OF=False):
    try:
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == 'x':
                    if isinstance(node.slice, ast.Num):
                        idx = node.slice.n
                        if idx < 0 or idx >= n_var:
                            return False, f"Index {idx} out of bounds for {n_var} variables."
                    else:
                        return False, "Dynamic indexing (e.g., x[i]) not supported."
                elif allow_OF and isinstance(node.value, ast.Name) and node.value.id == 'OF':
                    continue
                else:
                    return False, "Only 'x' and 'OF' (if allowed) are valid variables."
        return True, ""
    except Exception as e:
        return False, f"Invalid expression: {str(e)}"

def get_interactive_input():
    n_cont = get_int_input("Enter number of continuous variables: ", 0)
    n_disc = get_int_input("Enter number of discrete variables: ", 0)
    n_bin = 0
    n_var = n_cont + n_disc + n_bin

    if n_var > 15:
        print("Total number of variables must not exceed 15.")
        sys.exit(1)

    pop_size = get_int_input("Enter population size: ", 1, 500)
    n_elite = get_int_input(f"Enter number of elite preservations (0 to {pop_size}): ", 0, pop_size)
    n_gen = get_int_input("Enter the maximum number of generations: ", 1, 500)
    round_decimals = get_int_input("How many decimal places should results be rounded to? ", 0)
    n_runs = get_int_input("How many times do you want to run the algorithm? ", 1)

    xl_cont = []
    xu_cont = []
    print("\nEnter bounds for continuous variables:")
    for i in range(n_cont):
        xl_cont.append(get_float_input(f"  Continuous Variable {i+1} Lower Bound: "))
        xu_cont.append(get_float_input(f"  Continuous Variable {i+1} Upper Bound: "))

    print("\nFor each discrete variable, specify how the values should be entered:")
    print("- Type 'lc' for Least Count based range (e.g., 1 to 10 with LC 1)")
    print("- Type 'set' to enter a custom list of values")
    disc_values = []
    disc_bounds = []
    disc_lc_flags = []
    disc_lcs = []

    for i in range(n_disc):
        while True:
            mode = input(f"  Discrete Variable {i+1}: Enter 'lc' or 'set': ").strip().lower()
            if mode == 'lc':
                lb = get_float_input("    Lower Bound: ")
                ub = get_float_input("    Upper Bound: ")
                lc = get_float_input("    Least Count: ")
                if lc <= 0:
                    print("    Least Count must be positive.")
                    continue
                values = list(np.arange(lb, ub + lc, lc))
                if not values:
                    print("    No values generated. Check bounds and least count.")
                    continue
                disc_values.append(values)
                disc_bounds.append((lb, ub))
                disc_lc_flags.append(True)
                disc_lcs.append(lc)
                break
            elif mode == 'set':
                try:
                    values = list(map(float, input("    Enter comma-separated values: ").split(",")))
                    if not values:
                        print("    List cannot be empty.")
                        continue
                    disc_values.append(sorted(values))
                    disc_bounds.append((min(values), max(values)))
                    disc_lc_flags.append(False)
                    disc_lcs.append(None)
                    break
                except ValueError:
                    print("    Invalid input. Please enter numbers separated by commas.")
            else:
                print("    Invalid option. Please enter 'lc' or 'set'.")

    xl = np.array(xl_cont + [b[0] for b in disc_bounds] + [0] * n_bin)
    xu = np.array(xu_cont + [b[1] for b in disc_bounds] + [1] * n_bin)

    print("\nDo you want to solve Standard problems or User-defined Functions?")
    ans = input("Enter 'S' for Standard Problems or 'U' for User-defined Problems: ").strip().upper()
    if ans == 'U':
        if n_var < 1:
            print("Error: User-defined problem requires at least 1 variable for the constraint 'x[0] > 0'.")
            sys.exit(1)
        obj_func_file = input("Enter the path to obj_func.py: ").strip()
        objective_function = load_objective_function(obj_func_file, n_var)
        constraints = ["x[0] > 0", "OF < 10"]
        if len(constraints) > 10:
            print("Maximum of 10 constraints allowed.")
            sys.exit(1)
        for constr in constraints:
            valid, error = validate_expression(constr.split('>')[0] if '>' in constr else
                                              constr.split('<')[0] if '<' in constr else constr,
                                              n_var, allow_OF=True)
            if not valid:
                print(f"Error in constraint: {error}")
                sys.exit(1)
    else:
        problem_name = input("Enter the standard problem name (e.g., 'g1'): ").strip()
        try:
            problem = pymoo_get_problem(problem_name)
            objective_function = problem.evaluate
            constraints = []
            if problem.n_constr > 0:
                print(f"Warning: Standard problem '{problem_name}' has constraints, but they are handled internally by pymoo.")
        except Exception as e:
            print(f"Error: Invalid standard problem '{problem_name}'. {str(e)}")
            sys.exit(1)

    print("\n" + "="*60)
    print("REPORT GENERATION OPTIONS")
    print("="*60)

    while True:
        generate_report = input("Do you want to generate a report? (yes/no): ").strip().lower()
        if generate_report in ['yes', 'y', 'no', 'n']:
            break
        print("Invalid input. Please enter 'yes' or 'no'.")

    report_options = {
        'generate_report': generate_report in ['yes', 'y'],
        'population_wise': False,
        'generation_wise': False,
        'run_wise': False
    }

    if report_options['generate_report']:
        print("\nSelect ONE report type:")
        print("1. Population wise report for each generation in a run (Note: The program will be slow)")
        print("2. Generation wise report for each run")
        print("3. Run wise report")
    
        while True:
            try:
                choice = int(input("Enter your choice (1, 2, or 3): ").strip())
                if choice not in [1, 2, 3]:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    continue
                if choice == 1:
                    report_options['population_wise'] = True
                    print("Selected: Population wise report (each generation, each run)")
                    print("Note: This will significantly slow down the program.")
                elif choice == 2:
                    report_options['generation_wise'] = True
                    print("Selected: Generation wise report (final generation per run)")
                elif choice == 3:
                    report_options['run_wise'] = True
                    print("Selected: Run wise report (best solution per run)")
                break
            except ValueError:
                print("Invalid input. Please enter a number (1, 2, or 3).")

    return {
        'n_cont': n_cont, 'n_disc': n_disc, 'n_bin': n_bin, 'n_var': n_var,
        'pop_size': pop_size, 'n_elite': n_elite, 'n_gen': n_gen, 'round_decimals': round_decimals,
        'n_runs': n_runs, 'xl_cont': xl_cont, 'xu_cont': xu_cont,
        'disc_values': disc_values, 'disc_bounds': disc_bounds,
        'disc_lc_flags': disc_lc_flags, 'disc_lcs': disc_lcs,
        'xl': xl.tolist(), 'xu': xu.tolist(), 'objective_function': objective_function,
        'constraints': constraints, 'report_options': report_options
    }

def get_text_input():
    while True:
        filename = input("Enter the input text file path: ").strip()
        if not os.path.exists(filename):
            print("File does not exist. Please enter a valid file path.")
            continue
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            
            print("=== Raw File Contents ===")
            for i, line in enumerate(lines, 1):
                print(f"Line {i}: {line}")
            print("========================")

            min_lines = 7 + 2 * int(lines[0].split('=')[1].strip()) + 2 * int(lines[1].split('=')[1].strip())
            if len(lines) < min_lines:
                print("Invalid text file: Too few lines.")
                continue
            
            try:
                n_cont = int(lines[0].split('=')[1].strip())
                n_disc = int(lines[1].split('=')[1].strip())
                pop_size = int(lines[2].split('=')[1].strip())
                n_elite = int(lines[3].split('=')[1].strip())
                n_gen = int(lines[4].split('=')[1].strip())
                round_decimals = int(lines[5].split('=')[1].strip())
                n_runs = int(lines[6].split('=')[1].strip())
            except (IndexError, ValueError):
                print("Invalid text file: Error parsing basic parameters.")
                continue

            n_bin = 0
            n_var = n_cont + n_disc + n_bin
            if n_var > 15:
                print("Total number of variables must not exceed 15.")
                sys.exit(1)
            if not (1 <= pop_size <= 500):
                print("Population size must be between 1 and 500.")
                continue
            if not (0 <= n_elite <= pop_size):
                print(f"Number of elite preservations must be between 0 and {pop_size}.")
                continue
            if not (1 <= n_gen <= 500):
                print("Number of generations must be between 1 and 500.")
                continue
            if round_decimals < 0:
                print("Round decimals must be non-negative.")
                continue
            if n_runs < 1:
                print("Number of runs must be positive.")
                continue

            xl_cont = []
            xu_cont = []
            line_idx = 7
            try:
                for i in range(n_cont):
                    xl_cont.append(float(lines[line_idx].split('=')[1].strip()))
                    line_idx += 1
                for i in range(n_cont):
                    xu_cont.append(float(lines[line_idx].split('=')[1].strip()))
                    line_idx += 1
                if len(xl_cont) != n_cont or len(xu_cont) != n_cont:
                    print("Continuous variable bounds length mismatch.")
                    continue
            except (IndexError, ValueError):
                print("Invalid text file: Error parsing continuous variable bounds.")
                continue

            disc_values = []
            disc_bounds = []
            disc_lc_flags = []
            disc_lcs = []
            for i in range(n_disc):
                try:
                    mode = lines[line_idx].split('=')[1].strip().lower()
                    line_idx += 1
                    if mode == 'lc':
                        lb = float(lines[line_idx].split('=')[1].strip())
                        line_idx += 1
                        ub = float(lines[line_idx].split('=')[1].strip())
                        line_idx += 1
                        lc = float(lines[line_idx].split('=')[1].strip())
                        line_idx += 1
                        if lc <= 0:
                            print("Least count must be positive.")
                            sys.exit(1)
                        values = list(np.arange(lb, ub + lc, lc))
                        if not values:
                            print("No values generated for discrete variable.")
                            sys.exit(1)
                        disc_values.append(values)
                        disc_bounds.append((lb, ub))
                        disc_lc_flags.append(True)
                        disc_lcs.append(lc)
                    elif mode == 'set':
                        n_values = int(lines[line_idx].split('=')[1].strip())
                        line_idx += 1
                        values = []
                        for j in range(n_values):
                            values.append(float(lines[line_idx].split('=')[1].strip()))
                            line_idx += 1
                        if not values:
                            print("Discrete variable value set cannot be empty.")
                            sys.exit(1)
                        disc_values.append(sorted(values))
                        disc_bounds.append((min(values), max(values)))
                        disc_lc_flags.append(False)
                        disc_lcs.append(None)
                    else:
                        print("Invalid discrete variable type. Use 'lc' or 'set'.")
                        sys.exit(1)
                except (IndexError, ValueError):
                    print("Invalid text file: Error parsing discrete variable.")
                    continue

            try:
                ans = lines[line_idx].split('=')[1].strip().upper()
                line_idx += 1
                objective_function = None
                constraints = []
                if ans == 'S':
                    problem_name = lines[line_idx].split('=')[1].strip()
                    line_idx += 1
                    try:
                        problem = pymoo_get_problem(problem_name)
                        objective_function = problem.evaluate
                        constraints = []
                        if problem.n_constr > 0:
                            print(f"Warning: Standard problem '{problem_name}' has constraints, but they are handled internally by pymoo.")
                    except Exception as e:
                        print(f"Error: Invalid standard problem '{problem_name}'. {str(e)}")
                        sys.exit(1)
                else:
                    obj_func_file = lines[line_idx].split('=')[1].strip()
                    line_idx += 1
                    objective_function = load_objective_function(obj_func_file, n_var)
                    if line_idx < len(lines):
                        constraint_input = lines[line_idx].split('=')[1].strip()
                        constraints = [c.strip() for c in constraint_input.split(';') if c.strip()]
                        if len(constraints) > 10:
                            print("Maximum of 10 constraints allowed.")
                            continue
                    for constr in constraints:
                        valid, error = validate_expression(constr.split('>')[0] if '>' in constr else
                                                          constr.split('<')[0] if '<' in constr else constr,
                                                          n_var, allow_OF=True)
                        if not valid:
                            print(f"Invalid constraint: {error}")
                            sys.exit(1)
            except IndexError:
                print("Invalid text file: Missing problem type or objective function details.")
                continue

            xl = np.array(xl_cont + [b[0] for b in disc_bounds] + [0] * n_bin)
            xu = np.array(xu_cont + [b[1] for b in disc_bounds] + [1] * n_bin)

            print("=== Parsed Input File Data ===")
            print("Number of continuous variables:", n_cont)
            print("Number of discrete variables:", n_disc)
            print("Number of binary variables:", n_bin)
            print("Total number of variables:", n_var)
            print("Population size:", pop_size)
            print("Number of elite preservations:", n_elite)
            print("Number of generations:", n_gen)
            print("Decimal places for rounding:", round_decimals)
            print("Number of runs:", n_runs)
            print("Continuous variable lower bounds:", xl_cont)
            print("Continuous variable upper bounds:", xu_cont)
            for i in range(n_disc):
                print(f"Discrete variable {i+1}:")
                print(f"  Type: {'lc' if disc_lc_flags[i] else 'set'}")
                print(f"  Lower bound: {disc_bounds[i][0]}")
                print(f"  Upper bound: {disc_bounds[i][1]}")
                if disc_lc_flags[i]:
                    print(f"  Least count: {disc_lcs[i]}")
                print(f"  Values: {disc_values[i]}")
            print("Objective function:", objective_function)
            print("Constraints:", constraints if constraints else "None")
            print("======================")

            print("\n" + "="*60)
            print("REPORT GENERATION OPTIONS")
            print("="*60)

            while True:
                generate_report = input("Do you want to generate a report? (yes/no): ").strip().lower()
                if generate_report in ['yes', 'y', 'no', 'n']:
                    break
                print("Invalid input. Please enter 'yes' or 'no'.")

            report_options = {
                'generate_report': generate_report in ['yes', 'y'],
                'population_wise': False,
                'generation_wise': False,
                'run_wise': False
            }

            if report_options['generate_report']:
                print("\nSelect ONE report type:")
                print("1. Population wise report for each generation in a run (Note: The program will be slow)")
                print("2. Generation wise report for each run")
                print("3. Run wise report")
    
                while True:
                    try:
                        choice = int(input("Enter your choice (1, 2, or 3): ").strip())
                        if choice not in [1, 2, 3]:
                            print("Invalid choice. Please enter 1, 2, or 3.")
                            continue
                        if choice == 1:
                            report_options['population_wise'] = True
                            print("Selected: Population wise report (each generation, each run)")
                            print("Note: This will significantly slow down the program.")
                        elif choice == 2:
                            report_options['generation_wise'] = True
                            print("Selected: Generation wise report (final generation per run)")
                        elif choice == 3:
                            report_options['run_wise'] = True
                            print("Selected: Run wise report (best solution per run)")
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number (1, 2, or 3).")

            return {
                'n_cont': n_cont, 'n_disc': n_disc, 'n_bin': n_bin, 'n_var': n_var,
                'pop_size': pop_size, 'n_elite': n_elite, 'n_gen': n_gen, 'round_decimals': round_decimals,
                'n_runs': n_runs, 'xl_cont': xl_cont, 'xu_cont': xu_cont,
                'disc_values': disc_values, 'disc_bounds': disc_bounds,
                'disc_lc_flags': disc_lc_flags, 'disc_lcs': disc_lcs,
                'xl': xl.tolist(), 'xu': xu.tolist(), 'objective_function': objective_function,
                'constraints': constraints, 'report_options': report_options
            }
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            continue

# Prompt user for input method
while True:
    input_method = input("Choose input method: 'interactive' or 'file': ").strip().lower()
    if input_method in ['interactive', 'file']:
        break
    print("Invalid choice. Please enter 'interactive' or 'file'.")

input_data = get_interactive_input() if input_method == 'interactive' else get_text_input()

# Unpack input data
n_cont = input_data['n_cont']
n_disc = input_data['n_disc']
n_bin = input_data['n_bin']
n_var = input_data['n_var']
pop_size = input_data['pop_size']
n_elite = input_data['n_elite']
n_gen = input_data['n_gen']
round_decimals = input_data['round_decimals']
n_runs = input_data['n_runs']
xl_cont = input_data['xl_cont']
xu_cont = input_data['xu_cont']
disc_values = input_data['disc_values']
disc_bounds = input_data['disc_bounds']
disc_lc_flags = input_data['disc_lc_flags']
disc_lcs = input_data['disc_lcs']
xl = np.array(input_data['xl'])
xu = np.array(input_data['xu'])
objective_function = input_data['objective_function']
constraints = input_data['constraints']
report_options = input_data['report_options']

# ========== SAMPLING ==========
class CustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        sampler = LatinHypercubeSampling()
        raw = sampler._do(problem, n_samples, **kwargs)
        scaled = problem.xl + (problem.xu - problem.xl) * raw

        for i in range(n_disc):
            disc_idx = n_cont + i
            for k in range(n_samples):
                Vk = scaled[k, disc_idx]
                scaled[k, disc_idx] = snap_discrete_value(
                    Vk, *disc_bounds[i],
                    lc=disc_lcs[i] if disc_lc_flags[i] else None,
                    disc_values=None if disc_lc_flags[i] else disc_values[i]
                )

        for i in range(n_bin):
            bin_idx = n_cont + n_disc + i
            scaled[:, bin_idx] = np.random.randint(0, 2, size=scaled.shape[0])

        return scaled

# ========== SURVIVAL WITH OBJECTIVE PARAMETER NICHING AND ELITE PRESERVATION ==========
class FitnessSurvival(Survival):
    def __init__(self, sigma_share=0.2, n_elite=0):
        super().__init__()
        self.filter_infeasible = False
        self.sigma_share = sigma_share
        self.n_elite = n_elite

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F, cv = pop.get("F", "cv")
        n = len(pop)
        sharing = np.ones(n)

        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(F[i, 0] - F[j, 0])
                if dist < self.sigma_share:
                    sh = 1 - dist / self.sigma_share
                    sharing[i] += sh
                    sharing[j] += sh

        F_shared = F[:, 0] / sharing
        S = np.lexsort([F_shared, cv])
        
        # Preserve top n_elite individuals
        elite_indices = S[:self.n_elite]
        remaining_indices = S[self.n_elite:n_survive]
        final_indices = np.concatenate([elite_indices, remaining_indices])
        
        pop.set("rank", np.argsort(S))
        return pop[final_indices]

def comp_by_cv_and_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)
        else:
            S[i] = compare(a, pop[a].F, b, pop[b].F, method='smaller_is_better', return_random_if_equal=True)
    return S[:, None].astype(int)

# ========== ALGORITHM SETUP ==========
termination = get_termination("n_gen", n_gen)
crossover = SBX(prob=0.8, eta=15)
selection = TournamentSelection(func_comp=comp_by_cv_and_fitness, pressure=2)

class CustomMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        eta = 20
        prob_var = 1.0 / problem.n_var
        for i in range(len(X)):
            for j in range(problem.n_var):
                if np.random.rand() < prob_var:
                    if j < n_cont:
                        delta = (problem.xu[j] - problem.xl[j]) * 0.1
                        mutated_value = X[i, j] + np.random.uniform(-delta, delta)
                        mutated_value = np.clip(mutated_value, problem.xl[j], problem.xu[j])
                    elif j < n_cont + n_disc:
                        idx = j - n_cont
                        delta = (problem.xu[j] - problem.xl[j]) * 0.1
                        mutated_value = X[i, j] + np.random.uniform(-delta, delta)
                        mutated_value = snap_discrete_value(
                            mutated_value, *disc_bounds[idx],
                            lc=disc_lcs[idx] if disc_lc_flags[idx] else None,
                            disc_values=None if disc_lc_flags[idx] else disc_values[idx]
                        )
                    else:
                        mutated_value = 1 if X[i, j] == 0 else 0
                    X[i, j] = mutated_value
        return X

class CustomProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_var, n_obj=1, n_constr=len(constraints), xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        G = []
        safe_dict = {
            'np': np, 'math': math, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt
        }
        epsilon = 1e-6
        for x in X:
            local_dict = safe_dict.copy()
            local_dict['x'] = x
            try:
                OF = objective_function(x)
                F.append(OF)
                local_dict['OF'] = OF

                g = []
                for constr in constraints:
                    if ">=" in constr:
                        left, right = constr.split(">=")
                        g_expr = f"({right}) - ({left})"
                        g_value = eval(g_expr, {}, local_dict)
                        g.append(g_value)
                    elif ">" in constr:
                        left, right = constr.split(">")
                        g_expr = f"({right}) - ({left}) + {epsilon}"
                        g_value = eval(g_expr, {}, local_dict)
                        g.append(g_value)
                    elif "<=" in constr:
                        left, right = constr.split("<=")
                        g_expr = f"({left}) - ({right})"
                        g_value = eval(g_expr, {}, local_dict)
                        g.append(g_value)
                    elif "<" in constr:
                        left, right = constr.split("<")
                        g_expr = f"({left}) - ({right}) - {epsilon}"
                        g_value = eval(g_expr, {}, local_dict)
                        g.append(g_value)
                    else:
                        print(f"Warning: Constraint '{constr}' not recognized, treated as violated.")
                        g.append(1)
                G.append(g)
            except Exception as e:
                print(f"Error evaluating objective or constraint: {e}")
                F.append(np.inf)
                G.append([1] * len(constraints))

        out["F"] = np.array(F)
        out["G"] = np.array(G)

# ========== MAIN ALGORITHM LOOP ==========
report_data = {
    'population_data': [],
    'generation_data': [],
    'run_data': []
}

all_fitness_over_time = []

for run in range(n_runs):
    print(f"\n==== Run {run+1} ====")
    algorithm = GeneticAlgorithm(
        pop_size=pop_size,
        sampling=CustomSampling(),
        crossover=crossover,
        mutation=CustomMutation(),
        survival=FitnessSurvival(sigma_share=0.1, n_elite=n_elite),
        selection=selection,
        eliminate_duplicates=True
    )
    res = minimize(
        CustomProblem(),
        algorithm,
        termination,
        seed=run,
        save_history=True,
        verbose=False
    )
    
    if report_options['population_wise']:
        run_population_data = []
        for gen_idx, gen_history in enumerate(res.history):
            gen_pop = gen_history.pop
            X_gen = gen_pop.get("X")
            F_gen = gen_pop.get("F")
            
            X_gen_rounded = X_gen.copy()
            if n_cont > 0:
                X_gen_rounded[:, :n_cont] = np.round(X_gen[:, :n_cont], round_decimals)
            F_gen_rounded = np.round(F_gen, round_decimals)
            
            gen_data = []
            for i in range(len(X_gen_rounded)):
                gen_data.append({
                    'individual': i+1,
                    'variables': X_gen_rounded[i].tolist(),
                    'objective': F_gen_rounded[i][0]
                })
            
            run_population_data.append({
                'generation': gen_idx,
                'population': gen_data
            })
        
        report_data['population_data'].append({
            'run': run+1,
            'generations': run_population_data
        })
    
    print(f"\nGeneration 0 Population for Run {run+1}:")
    gen0_pop = res.history[0].pop
    X_gen0 = gen0_pop.get("X")
    F_gen0 = gen0_pop.get("F")
    X_gen0_rounded = X_gen0.copy()
    if n_cont > 0:
        X_gen0_rounded[:, :n_cont] = np.round(X_gen0[:, :n_cont], round_decimals)
    F_gen0_rounded = np.round(F_gen0, round_decimals)
    for i in range(len(X_gen0_rounded)):
        print(f"Population {i+1}: X = {X_gen0_rounded[i]}, OF = {F_gen0_rounded[i][0]}")

    print(f"\nFinal Population for Run {run+1}:")
    X = res.pop.get("X")
    X_rounded = X.copy()
    if n_cont > 0:
        X_rounded[:, :n_cont] = np.round(X[:, :n_cont], round_decimals)
    F = np.round(res.pop.get("F"), round_decimals)
    for i in range(len(X_rounded)):
        print(f"Population {i+1}: X = {X_rounded[i]}, OF = {F[i][0]}")
    
    best = np.argmin(F[:, 0])
    best_solution = X_rounded[best]
    best_objective = F[best][0]
    
    print(f"\nBest Solution: {best_solution}")
    print(f"Best Objective Function value: {best_objective}")
    print(f"Total Generations Run: {len(res.history)}")
    
    if report_options['generation_wise']:
        final_gen_data = []
        for i in range(len(X_rounded)):
            final_gen_data.append({
                'individual': i+1,
                'variables': X_rounded[i].tolist(),
                'objective': F[i][0]
            })
        
        report_data['generation_data'].append({
            'run': run+1,
            'final_generation': final_gen_data,
            'best_individual': {
                'variables': best_solution.tolist(),
                'objective': best_objective
            },
            'total_generations': len(res.history)
        })
    
    if report_options['run_wise']:
        report_data['run_data'].append({
            'run': run+1,
            'best_solution': best_solution.tolist(),
            'best_objective': best_objective,
            'total_generations': len(res.history)
        })
    
    fitness_over_time = [round(gen.opt.get("F")[0][0], round_decimals) for gen in res.history]
    all_fitness_over_time.append(fitness_over_time)

# ========== CONVERGENCE PLOT WITH MARKERS ==========
convergence_analysis = []
for i, fitness_data in enumerate(all_fitness_over_time):
    run_analysis = {
        'run': i + 1,
        'fitness_history': fitness_data,
        'convergence_point': None,
        'no_change_generations': 0
    }
    best_so_far = float('inf')
    last_improvement_gen = 0
    for gen, fitness in enumerate(fitness_data):
        if fitness < best_so_far:
            best_so_far = fitness
            last_improvement_gen = gen
    run_analysis['convergence_point'] = last_improvement_gen
    run_analysis['no_change_generations'] = len(fitness_data) - 1 - last_improvement_gen
    convergence_analysis.append(run_analysis)

plt.figure(figsize=(10, 5))
line_styles = ['--', '-.', ':', '-']
colors = ['red', 'blue', 'green', 'purple', 'orange']
for run, fitness in enumerate(all_fitness_over_time):
    plt.plot(fitness, marker='o', linestyle='-', label=f'Run {run+1}')
    conv_point = convergence_analysis[run]['convergence_point']
    plt.axvline(x=conv_point, color=colors[run % len(colors)], linestyle=line_styles[run % len(line_styles)],
                label=f'Convergence Run {run+1} (Gen {conv_point})', alpha=0.7)
plt.xlabel('Generation')
plt.ylabel('Best Objective Function Value')
plt.title('Convergence Plot of Real Coded Genetic Algorithm with Convergence Points')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ========== GENERATE REPORTS ==========
if report_options['generate_report']:
    print("\n" + "="*60)
    print("GENERATING REPORT")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n" + "="*80)
    print(timestamp)
    print("\n" + "="*80)
    
    if report_options['population_wise']:
        filename = f"population_wise_report_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("POPULATION WISE REPORT FOR EACH GENERATION IN A RUN\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Problem Parameters:\n")
            f.write(f"  - Continuous Variables: {n_cont}\n")
            f.write(f"  - Discrete Variables: {n_disc}\n")
            f.write(f"  - Population Size: {pop_size}\n")
            f.write(f"  - Number of Elite Preservations: {n_elite}\n")
            f.write(f"  - Maximum Generations: {n_gen}\n")
            f.write(f"  - Number of Runs: {n_runs}\n")
            f.write(f"  - Objective Function: {objective_function}\n")
            f.write(f"  - Constraints: {constraints}\n")
            f.write("="*80 + "\n\n")
            
            for run_data in report_data['population_data']:
                f.write(f"RUN {run_data['run']}\n")
                f.write("-" * 40 + "\n")
                
                for gen_data in run_data['generations']:
                    f.write(f"\nGeneration {gen_data['generation']}:\n")
                    for individual in gen_data['population']:
                        f.write(f"  Population {individual['individual']}: ")
                        f.write(f"Variables = {individual['variables']}, ")
                        f.write(f"Objective = {individual['objective']}\n")
                
                f.write("\n" + "="*80 + "\n")
        
        print("Generating population vs objective function plots for each generation...")
        for run_data in report_data['population_data']:
            run_num = run_data['run']
            generations = run_data['generations']
            
            n_gens = len(generations)
            cols = min(4, n_gens)
            rows = (n_gens + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            fig.suptitle(f'Population vs Objective Function - Run {run_num}', fontsize=16)
            
            if n_gens == 1:
                axes = np.array([axes])
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for gen_idx, gen_data in enumerate(generations):
                individuals = [ind['individual'] for ind in gen_data['population']]
                objectives = [ind['objective'] for ind in gen_data['population']]
                
                ax = axes[gen_idx] if gen_idx < len(axes) else axes[0]
                ax.bar(individuals, objectives, alpha=0.7, color='skyblue', edgecolor='navy')
                ax.set_title(f'Generation {gen_data["generation"]}')
                ax.set_xlabel('Individual')
                ax.set_ylabel('Objective Function Value')
                ax.grid(True, alpha=0.3)
            
            for i in range(n_gens, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_filename = f"population_plots_run_{run_num}_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Population plots for Run {run_num} saved as: {plot_filename}")
        
        print(f"✓ Population wise report saved as: {filename}")
    
    elif report_options['generation_wise']:
        filename = f"generation_wise_report_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GENERATION WISE REPORT FOR EACH RUN\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Problem Parameters:\n")
            f.write(f"  - Continuous Variables: {n_cont}\n")
            f.write(f"  - Discrete Variables: {n_disc}\n")
            f.write(f"  - Population Size: {pop_size}\n")
            f.write(f"  - Number of Elite Preservations: {n_elite}\n")
            f.write(f"  - Maximum Generations: {n_gen}\n")
            f.write(f"  - Number of Runs: {n_runs}\n")
            f.write(f"  - Objective Function: {objective_function}\n")
            f.write(f"  - Constraints: {constraints}\n")
            f.write("="*80 + "\n\n")
            
            f.write("CONVERGENCE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for analysis in convergence_analysis:
                f.write(f"Run {analysis['run']}:\n")
                f.write(f"  Last improvement at generation: {analysis['convergence_point']}\n")
                f.write(f"  Objective function stops changing after generation: {analysis['convergence_point']}\n")
                f.write(f"  No change for: {analysis['no_change_generations']} generations\n")
                if analysis['convergence_point'] < len(analysis['fitness_history']) - 1:
                    f.write(f"  Converged objective value: {analysis['fitness_history'][analysis['convergence_point']]:.{round_decimals}f}\n")
                else:
                    f.write(f"  Converged objective value: {analysis['fitness_history'][-1]:.{round_decimals}f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n\n")
            
            for run_data in report_data['generation_data']:
                f.write(f"RUN {run_data['run']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Generations: {run_data['total_generations']}\n\n")
                
                conv_info = convergence_analysis[run_data['run'] - 1]
                f.write(f"Best Population for Generation {conv_info['convergence_point']}:\n")
                for individual in run_data['final_generation']:
                    f.write(f"  Population {individual['individual']}: ")
                    f.write(f"Variables = {individual['variables']}, ")
                    f.write(f"Objective = {individual['objective']:.{round_decimals}f}\n")
                
                f.write(f"\nBest Solution in Run {run_data['run']}:\n")
                f.write(f"  Variables: {run_data['best_individual']['variables']}\n")
                f.write(f"  Objective: {run_data['best_individual']['objective']:.{round_decimals}f}\n")
                
                f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Generation wise report saved as: {filename}")
    
    elif report_options['run_wise']:
        filename = f"run_wise_report_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RUN WISE REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Problem Parameters:\n")
            f.write(f"  - Continuous Variables: {n_cont}\n")
            f.write(f"  - Discrete Variables: {n_disc}\n")
            f.write(f"  - Population Size: {pop_size}\n")
            f.write(f"  - Number of Elite Preservations: {n_elite}\n")
            f.write(f"  - Maximum Generations: {n_gen}\n")
            f.write(f"  - Number of Runs: {n_runs}\n")
            f.write(f"  - Objective Function: {objective_function}\n")
            f.write(f"  - Constraints: {constraints}\n")
            f.write("="*80 + "\n\n")
            
            f.write("SUMMARY OF ALL RUNS:\n")
            f.write("-" * 40 + "\n")
            
            best_overall = min(report_data['run_data'], key=lambda x: x['best_objective'])
            
            for run_data in report_data['run_data']:
                f.write(f"Run {run_data['run']}:\n")
                f.write(f"  Best Variables: {run_data['best_solution']}\n")
                f.write(f"  Best Objective: {run_data['best_objective']:.{round_decimals}f}\n")
                f.write(f"  Total Generations: {run_data['total_generations']}\n\n")
            
            f.write("OVERALL BEST SOLUTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best Run: {best_overall['run']}\n")
            f.write(f"Best Variables: {best_overall['best_solution']}\n")
            f.write(f"Best Objective: {best_overall['best_objective']:.{round_decimals}f}\n")
            f.write(f"Generations Required: {best_overall['total_generations']}\n")
            
            f.write(f"\nSTATISTICAL SUMMARY:\n")
            f.write("-" * 40 + "\n")
            objectives = [run['best_objective'] for run in report_data['run_data']]
            f.write(f"Mean Objective: {np.mean(objectives):.{round_decimals}f}\n")
            f.write(f"Standard Deviation: {np.std(objectives):.{round_decimals}f}\n")
            f.write(f"Best Objective: {np.min(objectives):.{round_decimals}f}\n")
            f.write(f"Worst Objective: {np.max(objectives):.{round_decimals}f}\n")
        
        print("Generating bar plot for runs vs objective function values...")
        runs = [run['run'] for run in report_data['run_data']]
        objectives = [run['best_objective'] for run in report_data['run_data']]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(runs, objectives, color='lightcoral', edgecolor='darkred', alpha=0.7)
        
        for bar, obj in zip(bars, objectives):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(objectives)*0.01, 
                    f'{obj:.{round_decimals}f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Run Number', fontsize=12, fontweight='bold')
        plt.ylabel('Best Objective Function Value', fontsize=12, fontweight='bold')
        plt.title('Best Objective Function Value by Run', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        best_run_index = objectives.index(min(objectives))
        bars[best_run_index].set_color('gold')
        bars[best_run_index].set_edgecolor('orange')
        
        plt.legend(['Regular Runs', 'Best Run'], loc='upper right')
        
        plt.tight_layout()
        plot_filename = f"run_wise_barplot_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Run wise report saved as: {filename}")
        print(f"✓ Bar plot saved as: {plot_filename}")
    
    print("\nReport generation completed successfully!")
else:
    print("\nNo report generated.")