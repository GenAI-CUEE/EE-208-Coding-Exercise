 Our goal is to provide the code-based visualization to understand the idea behind some parts of the lectures in **Programming for EE 2102208** by *Suwichaya Suwanwimolkul*.

You can download course materials from [here](https://drive.google.com/drive/folders/1W5l3XV3gomwPGVU86zc30-jYs8YocZxH?usp=drive_link)

### Topics

- [Python basic command](Python_basic_commands/Readme.md)
  This folder contains the python scripts for each topic:

  | Topic | Script |
  |-------|--------|
  | `if`/`elif`/`else` combined with `try`/`except` for input validation | [Tutorial_ifelse_tryexcept.py](Python_basic_commands/Tutorial_ifelse_tryexcept.py) |
  | Basic list operations (`append`, indexing, iteration) | [Tutorial_list.py](Python_basic_commands/Tutorial_list.py) |
  | Debugging with `pdb`/`breakpoint()` | [tutorial_pdb.py](Python_basic_commands/tutorial_pdb.py) |
  | `try`/`except` for handling invalid conversions | [Tutorial_try_except.py](Python_basic_commands/Tutorial_try_except.py) |


- [Loop Tracer](Loop_Tracer/Readme.md)
  A short exercise in using Python's interactive debugger (`pdb`) to step through a `for` loop one iteration at a time, watching variables come into existence as the loop runs.

    <p align="center">
      <img src="Loop_Tracer/PPT.png" alt="PPT" width="500">
    </p>


- [Debug the Module](Debug_the_Module/Readme.md)
  A short exercise in using Python's interactive debugger (`pdb`) to find a bug that lives in an **imported module**, not your main script.

    <p align="center">
    <img src="Debug_the_Module/PPT.png" alt="PPT" width="500">
    </p>

  

- [Moving average](Moving_Average/Readme.md)
  This programming exercise asks you to implement a sliding-window **moving average** over a list of integers, and to track the **FLOP (floating-point operation) count** incurred while computing it.

    <p align="center">
      <img src="Moving_Average/PPT.png" alt="PPT" width="500">
    </p>


- [Matmul and GFLOP](Matmul_and_GFLOP/Readme.md) 
  This exercise asks you to implement matrix multiplication using three nested Python loops and derive the FLOP count for matrix multiplication from its algorithmic definition. 


- [Numpy tutorial 00-09](Numpy_tutorial-00-09/Readme.md)
  This contains examples of numpy lecture. This folder contains the python scripts for each topic:

  | # | Topic | Script |
  |---|-------|--------|
  | 00 | Vector shapes: row vs. column vectors | [tutorial_numpy_00_vector_shape.py](Numpy_tutorial-00-09/tutorial_numpy_00_vector_shape.py) |
  | 01 | Shape mismatches and broadcasting errors | [tutorial_numpy_01_shape.py](Numpy_tutorial-00-09/tutorial_numpy_01_shape.py) |
  | 02 | NumPy arrays vs. Python lists: reference vs. `.copy()` | [tutorial_numpy_02_numpy_and_list_copy.py](Numpy_tutorial-00-09/tutorial_numpy_02_numpy_and_list_copy.py) |
  | 03 | Matrix multiplication (`@`, `np.dot`, `np.matmul`) | [tutorial_numpy_03_matrix_multiply.py](Numpy_tutorial-00-09/tutorial_numpy_03_matrix_multiply.py) |
  | 04 | Random vectors and matrices (`np.random.randint`) | [tutorial_numpy_04_random_vector_matrix.py](Numpy_tutorial-00-09/tutorial_numpy_04_random_vector_matrix.py) |
  | 04 | Zero and one matrices/vectors (`np.zeros`, `np.ones`) | [tutorial_numpy_04_zeros_ones.py](Numpy_tutorial-00-09/tutorial_numpy_04_zeros_ones.py) |
  | 05 | Identity and diagonal matrices (`np.eye`) | [tutorial_numpy_05_eyes.py](Numpy_tutorial-00-09/tutorial_numpy_05_eyes.py) |
  | 06 | `np.diag`: vector → diagonal matrix and back | [tutorial_numpy_06_diag_a_vs_diag_A.py](Numpy_tutorial-00-09/tutorial_numpy_06_diag_a_vs_diag_A.py) |
  | 07 | Lower/upper triangular matrices (`np.tril`, `np.triu`) | [tutorial_numpy_07_Random_Lower_Upper_Triangular.py](Numpy_tutorial-00-09/tutorial_numpy_07_Random_Lower_Upper_Triangular.py) |
  | 08 | Constructing a symmetric matrix | [tutorial_numpy_08_a_symetry_matrix.py](Numpy_tutorial-00-09/tutorial_numpy_08_a_symetry_matrix.py) |
  | 09 | Matrix inverse: general vs. diagonal matrix (`np.linalg.inv`) | [tutorial_numpy_09_Inverse_A_vs_Inverse_Diag.py](Numpy_tutorial-00-09/tutorial_numpy_09_Inverse_A_vs_Inverse_Diag.py) |



- [Student score numpy](Student_scores_numpy/Readme.md) 

    There are **2 bugs** in `student_scores.py`:

    - **Bug 1** — a NumPy array **shape** problem.
    - **Bug 2** — a `for`-loop `break` condition that's wrong.

    Your job: find and fix both bugs so `python report.py` runs cleanly
    and produces the correct report (see "Expected Output" below).

    <p align="center">
      <img src="Student_scores_numpy/PPT.png" alt="PPT" width="500">
    </p>
    


- [Python runtime profiling](Profiling_example/Readme.md)

    How to perform runtime profiling. \
    We provide collab example of profiling a 2D filtering example: 
    <a target="_blank" href="https://colab.research.google.com/github/GenAI-CUEE/EE-208-Coding-Exercise/blob/master/Profiling_example.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>

    <p align="center">
      <img src="Profiling_example/kitten_00000027.png" alt="PPT" width="200">
    </p>
