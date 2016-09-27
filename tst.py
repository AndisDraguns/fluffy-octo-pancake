import numpy as np
import itertools

side_length = 3
dimensions = 3
probability = 0.5

size = [side_length] * dimensions
size_minus = [side_length - 1] * dimensions
cube = np.zeros(size)

the_array = [0] * (side_length)
arrays = [0] * (dimensions)

counter = 0
while counter < side_length:
    the_array[counter] = counter
    counter += 1

counter = 0
while counter < dimensions:
    arrays[counter] = the_array
    counter += 1

mylist = (list(itertools.product(*arrays)))
# print(the_array)
# print(arrays)
# print(mylist)


def fill_random(cube, arrays):
    counter = 0
    while counter < len(mylist):
        a = np.random.uniform(0, 1)
        if a > probability:
            a = 1
        else:
            a = 0

        cube[mylist[counter]] = a
        counter += 1
    return cube

cube = fill_random(cube, arrays)

print(cube)
