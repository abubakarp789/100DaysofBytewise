def quicksort(arr):
    # Base case: if the array has 1 or fewer elements, it is already sorted
    if len(arr) <= 1:
        return arr
    
    # Choose a pivot element (in this case, the middle element)
    pivot = arr[0]
    
    # Partition the array into three parts: elements less than the pivot, equal to the pivot, and greater than the pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Recursively sort the left and right partitions, and combine them with the middle partition
    return quicksort(left) + middle + quicksort(right)

# Test the quicksort function with an example array
print(quicksort([3, 6, 8, 10, 1, 2, 15, 7, -4, 5, -9, 18, 5, 12, 9, -2]))

