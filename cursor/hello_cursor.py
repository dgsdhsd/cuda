#!/usr/bin/env python3
"""A tiny demo: SIMD-style vector add on CPU, printed as a mini CUDA tribute."""

from __future__ import annotations


def vector_add(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return [x + y for x, y in zip(a, b)]


def main() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    c = vector_add(a, b)

    print("cursor / hello_cursor.py")
    print(f"threads  = {len(a)}")
    print(f"A        = {a}")
    print(f"B        = {b}")
    print(f"C = A+B  = {c}")
    print("ok: python ran successfully")


if __name__ == "__main__":
    main()
