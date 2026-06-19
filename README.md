 Our goal is to provide the code-based visualization to understand the idea behind some parts of the lectures in **Programming for EE 2102208** by *Suwichaya Suwanwimolkul*.

### Topics

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
