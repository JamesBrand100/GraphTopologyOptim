import numpy as np

def normalize_array_and_reassign(arr):
    print(f"Inside function (before): id of arr is {id(arr)}")
    print(f"Inside function (before): arr is\n{arr}")
    
    # This creates a NEW array and the local 'arr' variable is now pointing to it
    arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)
    
    print(f"\nInside function (after): id of arr is {id(arr)}")
    print(f"Inside function (after): arr is\n{arr}")
    return arr

# The higher routine
original_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print(f"Outside function (before): id of original_data is {id(original_data)}")
print(f"Outside function (before): original_data is\n{original_data}")

# Pass the array to the function and capture the returned result
normalized_result = normalize_array_and_reassign(original_data)

print(f"\nOutside function (after): id of original_data is {id(original_data)}")
print(f"Outside function (after): original_data is\n{original_data}")

print(f"\nOutside function (after): id of normalized_result is {id(normalized_result)}")
print(f"Outside function (after): normalized_result is\n{normalized_result}")