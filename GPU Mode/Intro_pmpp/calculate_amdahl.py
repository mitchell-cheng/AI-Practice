import math

def calculate_amdahl(portion):
    """
    The function for calculating the theoretical speedup in latency according to Amdahl's law
    """
    return math.ceil(1 / (1 - portion))

# Usage example
portion_99 = 0.99
portion_95 = 0.95
portion_90 = 0.90
portion_75 = 0.75
portion_50 = 0.50
print("The theoretical maximum speedup for portion 95% is near: ", calculate_amdahl(portion_99)) # => 100
print("The theoretical maximum speedup for portion 95% is near: ", calculate_amdahl(portion_95)) # => 20
print("The theoretical maximum speedup for portion 90% is near: ", calculate_amdahl(portion_90)) # => 11
print("The theoretical maximum speedup for portion 75% is near: ", calculate_amdahl(portion_75)) # => 4
print("The theoretical maximum speedup for portion 50% is near: ", calculate_amdahl(portion_50)) # => 2