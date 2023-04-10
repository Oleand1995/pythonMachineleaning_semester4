import random

# The target string we want to generate
TARGET_STRING = "To be or not to be, that is the question."

# The characters we can use to generate new strings
POSSIBLE_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,."

# The function that generates a random string of the same length as the target string
def generate_random_string():
    return "".join([random.choice(POSSIBLE_CHARS) for _ in range(len(TARGET_STRING))])

# The function that calculates the fitness of a given string
def calculate_fitness(s):
    return sum([1 if s[i] == TARGET_STRING[i] else 0 for i in range(len(TARGET_STRING))])

# The hill climbing algorithm
def hill_climbing():
    current_string = generate_random_string()
    current_fitness = calculate_fitness(current_string)
    count = 0;
    while current_fitness < len(TARGET_STRING):
        new_string = list(current_string)
        # Randomly change one character in the current string
        new_string[random.randint(0, len(TARGET_STRING) - 1)] = random.choice(POSSIBLE_CHARS)
        new_fitness = calculate_fitness("".join(new_string))
        # If the new string is better, use it as the new current string
        if new_fitness > current_fitness:
            current_string = "".join(new_string)
            current_fitness = new_fitness
        count = count + 1;
        print(count, " ", current_string)
    print("Found the target string:", current_string)

# Run the hill climbing algorithm
hill_climbing()
