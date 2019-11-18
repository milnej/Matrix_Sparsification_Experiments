import numpy as np
import math as m

def sparse(A, error, n):
    A_hat = np.zeros((n,n))

    # removes very small entires
    for i in range(n):
        for j in range(n):
            if A[i][j] >= abs(error/(2*n)):
                A_hat[i][j] = A[i][j]

    print("A_hat:", A_hat)
    
    #calculate s
    fro = np.linalg.norm(A_hat, 'fro')**2
    sr = fro/np.linalg.norm(A_hat, 2)**2

    s = int((28*n*sr*np.log(m.sqrt(2)*n))/error**2)

    # create prob matrix
    probs = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            probs[i][j] = A_hat[i][j]**2/fro

    #partial sum vector
    partial = []
    partial_sum = 0
    for i in range(n):
        for j in range(n):
            partial_sum += probs[i][j]
            partial.append(partial_sum)

    # sample s values
    A_til = np.zeros((n,n))

    for t in range(s):
        r = np.random.uniform(0,1)

        for p in range(n**2):
            if partial[p] > r:
                i = int(p/n)
                j = p%n
                A_til[i][j] += A_hat[i][j]/(probs[i][j]*s)
                break

    return A_til

if __name__ == "__main__":

    np.random.seed(1234)

    n = 10
    max_val = 5
    error = .1

    A = np.random.rand(n,n)
    A *= max_val

    print("A:", A)
    A_til = sparse(A, error, n)
    print("A_til:", A_til)

    print(np.linalg.norm(A-A_til,2)/np.linalg.norm(A,2))
    print(error)