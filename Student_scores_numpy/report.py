"""
report.py
---------
Builds a class roster, computes final grades, and reports the first
failing student (if any), using `student_scores.py`.

Run with:
    python report.py

Expected behavior (once both bugs are fixed):
- Print each student's final grade as a plain number (not an array).
- Print the shape of the final grades array — should be (8,), i.e. 1D.
- Loop through students in roster order and report the FIRST student
  (if any) whose final grade is below 70.0.
- If no student is failing, print "No failing students found."
"""

import numpy as np
import pdb

from student_scores import (
    compute_final_scores,
    find_first_failing_student,
    PASSING_SCORE,
)

STUDENT_NAMES = [
    "Alice", "Bob", "Carla", "Dinesh",
    "Elena", "Farid", "Grace", "Hugo",
]

# Rows = students, columns = assignments (HW1, HW2, Midterm, Final)
scores = np.array([
    [85, 90, 88, 92],   # Alice
    [78, 82, 75, 80],   # Bob
    [92, 95, 91, 96],   # Carla
    [60, 65, 55, 62],   # Dinesh  <- should be the first failing student
    [88, 84, 90, 87],   # Elena
    [70, 68, 72, 71],   # Farid
    [95, 98, 94, 97],   # Grace
    [55, 60, 58, 63],   # Hugo    <- also failing, but AFTER Dinesh
])

# Assignment weights: HW1 15%, HW2 15%, Midterm 30%, Final 40%
weights = np.array([0.15, 0.15, 0.30, 0.40])


def main():
    final_scores = compute_final_scores(scores, weights)

    print("=== Final Grades ===") 
    print(f"final_scores.shape = {final_scores.shape}  (expected: (8,))")
    for name, score in zip(STUDENT_NAMES, final_scores): 
        print("%s: %.2f" % (name, score))
 

    print("\n=== First Failing Student (score < %.1f) ===" % PASSING_SCORE)
 
    idx, name = find_first_failing_student(final_scores, STUDENT_NAMES)
    if idx is None:
        print("No failing students found.")
    else: 
        print("First failing student: %s (index %d), score = %.2f" % (name, idx, final_scores[idx]))


if __name__ == "__main__":
    main()
