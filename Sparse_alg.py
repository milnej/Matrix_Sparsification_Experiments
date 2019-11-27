import numpy as np
import math as m

# Function that returns the number of samples required for sparse matrix
def s_calc(A, error, n):
    sr = stable_rank(A)
    return int((28*n*sr*np.log(m.sqrt(2)*n))/error**2)

# Function that compares the error of the sparse matrix to the expected error
# Return 1 on succcess and 0 on failure
def sparse_error_success(A, A_til, error):
    comparison = np.linalg.norm(A-A_til, 2)/np.linalg.norm(A,2)
    if comparison <= error:
        return 1
    return 0

# Helper function that computes the stable rank of a matrix A
def stable_rank(A):
    return np.linalg.norm(A, 'fro')**2 / np.linalg.norm(A, 2)**2

# Function that returns a sparse representation of a square n x n matrix A
def sparsify(A, error, n, s, trunc):
    # Set up matrix space for truncated matrix
    A_hat = np.zeros((n,n))

    # Compute epsilon from our expected error
    eps = error * np.linalg.norm(A,2)

    # Truncation: removes entries smaller than eps/2n
    if trunc: # Option to toggle off truncation for experimenting...
        for i in range(n):
            for j in range(n):
                if A[i][j] >= abs(eps/(2*n)):
                    A_hat[i][j] = A[i][j]
    
    # Calculate Frobenius norm
    fro = np.linalg.norm(A_hat, 'fro')**2

    # Create matrix that holds probabilities for selecting the entry at (i,j)
    probs = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # Probability distribution that favors larger entries
            probs[i][j] = A_hat[i][j]**2/fro

    # Partial sum vector that holds probability ranges
    # Landing in one of the ranges selects an entry (i,j)
    partial = []
    partial_sum = 0
    for i in range(n):
        for j in range(n):
            partial_sum += probs[i][j]
            partial.append(partial_sum)

    # Set up matrix space for sparse matrix via sampling
    A_til = np.zeros((n,n))

    # Sample s entries
    for t in range(s):
        # Randomly select an entry
        r = np.random.uniform(0,1)
        for p in range(n**2):
            if partial[p] > r:
                i = int(p/n)
                j = p%n
                # Add entry into sparse matrix after scaling it based on likelihood of selection
                A_til[i][j] += A_hat[i][j]/(probs[i][j]*s)
                break

    # Return sparse matrix
    return A_til

if __name__ == "__main__":

    ## Unused test case for sparsify function
    np.random.seed(1234)

    n = 10
    error = .1

    num_large = int(n**2/10)
    max_val = 5
    A = np.random.rand(n,n) * max_val
    for i in range(num_large):
        A[np.random.randint(0,n-1)][np.random.randint(0,n-1)] = max_val * n

    print("A:", A)
    A_til = sparsify(A, error, n, int(n**2), True)
    print("A_til:", A_til)

    print(np.linalg.norm(A-A_til,2)/np.linalg.norm(A,2))
    print(error)

    