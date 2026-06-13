"""
student_scores.py
------------------
A small module that computes final grades for a class, using NumPy for
the heavy lifting and a plain Python for-loop for one "early stopping"
search.

Used by `report.py`, which:
    1. Builds a (n_students, n_assignments) matrix of raw scores.
    2. Calls `compute_final_grades()` to get a weighted final grade per
       student (NumPy matrix math).
    3. Calls `find_first_failing_student()` to find the FIRST student
       (in roster order) whose final grade is below the passing
       threshold, using a for-loop with a break.

NOTE FOR STUDENTS: This file contains exactly TWO bugs:

  BUG 1 (in `compute_final_grades`): a NumPy SHAPE bug. A weights
        vector is combined with the scores matrix using the wrong
        shape/axis, causing either a crash or a silently wrong result.

  BUG 2 (in `find_first_failing_student`): a for-loop with a `break`
        statement. The break CONDITION is wrong, so the loop either
        breaks too early, too late, or finds the wrong student.

Your job: use a debugger (or print statements) to set a breakpoint
inside the for-loop, inspect shapes with `.shape`, and fix both bugs.
"""

import numpy as np


PASSING_SCORE = 70.0


def compute_final_scores(scores, weights):
    """
    Compute each student's final scores as a weighted average of their
    assignment scores.

    Parameters
    ----------
    scores : ndarray, shape (n_students, n_assignments)
        Raw scores for each student on each assignment (0-100).
    weights : ndarray, shape (n_assignments,)
        Weight for each assignment. Weights sum to 1.0.

    Returns
    -------
    ndarray, shape (n_students,)
        Final weighted grade for each student.
    """
    n_students, n_assignments = scores.shape

    # BUG 1: `weights` has shape (n_assignments,), but here it's reshaped
    # to (n_assignments, 1) -- a COLUMN vector -- before being used with
    # `scores` (shape (n_students, n_assignments)). This makes
    # `scores @ weights_col` produce shape (n_students, 1) instead of
    # (n_students,), and `final_grades[i]` later in `report.py` ends up
    # comparing a length-1 array instead of a plain float, which can
    # cause confusing comparisons in BUG 2.
    weights_col = weights.reshape(n_assignments, 1)
    final_scores = scores @ weights_col

    return final_scores


def find_first_failing_student(final_scores, student_names):
    """
    Find the first student (in roster order) whose final grade is
    below `PASSING_SCORE`.

    Parameters
    ----------
    final_scores : ndarray, shape (n_students,)
        Final weighted scores for each student.
    student_names : list of str, length n_students

    Returns
    -------
    tuple (int, str) or (None, None)
        (index, name) of the first failing student, or (None, None) if
        every student passed.
    """
    first_failing_index = None
    first_failing_name = None

    for i, name in enumerate(student_names):
        grade = final_scores[i]

        # >>> SET A BREAKPOINT ON THE NEXT LINE <<<
        # Inspect: i, name, grade, grade.shape (if it's an array),
        # and PASSING_SCORE. Is `grade` the type/shape you expect?

        # BUG 2: this condition is wrong. As written, it breaks out of
        # the loop on the very FIRST student regardless of their grade
        # (or, depending on how you "fix" it naively, it might never
        # break at all). Figure out the correct condition that stops
        # the loop at the FIRST student whose grade is below
        # PASSING_SCORE.
        if i >= 0:
            first_failing_index = i
            first_failing_name = name
            break

    return first_failing_index, first_failing_name
