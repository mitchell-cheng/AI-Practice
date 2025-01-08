# Notes on GPU-Mode lecture 2

Lecture 2 is about covering the chapter 1~3 from book _Programming Massively Parallel Processors_.

## Chapter 1

### Amdahl's law

> "the overall performance improvement gained by optimizing a single part of a system is limited by the fraction of time that the improved part is actually used". - Amdahl's law

- verbally: Amdahl's law suggests that for a given system, there exists a theoratical limit which the system can be optimized for the latency of the fixed workload taks
- visually: ![Amdahl's law - from wikipedia](./image.png)
- algebraically: `S ~= 1 / (1 - p)`
- numerically: If `95%` of the program can be parallelized, the theoratical maximum speedup using parallel computing would be `20` times
- algorithmically:

```python
def calculate_amdahl(portion):
    """
    The function for calculating the theoretical speedup in latency according to Amdahl's law
    """
    return math.ceil(1 / (1 - portion))

# Usage example
portion_95 = 0.95
print("The theoretical maximum speedup for portion 95% is near: ", calculate_amdahl(portion_95))
```

For many real applications, `99%` of the program can be parallelize and `>100x` speedups are expected.

# Reference

- [GPU Mode - Github](https://github.com/gpu-mode/lectures)
- [Amdahl's law - Wikipedia.org](https://en.wikipedia.org/wiki/Amdahl%27s_law)
