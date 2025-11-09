"""Helper function for Euclidean tensor contractions (Issue #28461)"""

from sympy.tensor.tensor import TensorIndexType, TensorHead, tensor_indices
from sympy import Symbol

def euclidean_contract(*tensors):
    """
    Contract tensors treating covariant and contravariant indices as equivalent.
    Automatically converts repeated indices to opposite signs for Einstein summation.
    """
    
    if len(tensors) == 0:
        return Symbol('1')
    
    if len(tensors) == 1:
        return tensors[0]
    
    # Collect all indices from all tensors
    from sympy.tensor.tensor import get_indices
    
    all_indices = []
    tensor_index_lists = []
    
    for tensor in tensors:
        indices = get_indices(tensor)
        tensor_index_lists.append(indices)
        all_indices.extend(indices)
    
    # Find repeated indices (potential contractions)
    index_counts = {}
    for idx in all_indices:
        # Use absolute value to treat i and -i as same index
        abs_idx = idx if idx.is_up else -idx
        if abs_idx not in index_counts:
            index_counts[abs_idx] = []
        index_counts[abs_idx].append(idx)
    
    # Find which indices need to be flipped
    replacements = [{} for _ in tensors]
    
    for abs_idx, occurrences in index_counts.items():
        if len(occurrences) == 2:
            # This is a contraction - ensure one is up, one is down
            idx1, idx2 = occurrences
            
            # If both have same orientation, flip the second one
            if idx1.is_up == idx2.is_up:
                # Find which tensor has the second occurrence
                count = 0
                for tensor_num, indices in enumerate(tensor_index_lists):
                    for pos, idx in enumerate(indices):
                        abs_test = idx if idx.is_up else -idx
                        if abs_test == abs_idx:
                            count += 1
                            if count == 2:
                                # Flip this index
                                replacements[tensor_num][idx] = -idx
                                break
                    if count == 2:
                        break
        elif len(occurrences) > 2:
            # Multiple contractions - alternate up/down
            for i, idx in enumerate(occurrences):
                if i > 0 and idx.is_up == occurrences[i-1].is_up:
                    # Find and flip
                    for tensor_num, indices in enumerate(tensor_index_lists):
                        if idx in indices:
                            replacements[tensor_num][idx] = -idx
                            break
    
    # Apply replacements
    modified_tensors = []
    for tensor, repl in zip(tensors, replacements):
        if repl:
            # Replace indices in tensor
            modified = tensor._replace_indices(repl)
            modified_tensors.append(modified)
        else:
            modified_tensors.append(tensor)
    
    # Now multiply them together
    result = modified_tensors[0]
    for tensor in modified_tensors[1:]:
        result = result * tensor
    
    return result


if __name__ == "__main__":
    # Test case for Issue #28461
    index_space = TensorIndexType('X')
    A = TensorHead('A', [index_space] * 2)
    B = TensorHead('B', [index_space] * 1)
    i, j = tensor_indices('i j', index_space)
    
    result = euclidean_contract(A(i, j), B(j))
    print(result)  # A(i, X_0)*B(-X_0)
