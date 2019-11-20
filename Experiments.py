from matplotlib import pyplot as plt
from Sparse_alg import *

if __name__ == '__main__':

    #np.random.seed(1234)

    # make matricies
    n = 10

    # high variance
    num_large = int(n**2/10)
    max_val = 5
    A = np.random.rand(n,n) * max_val
    for i in range(num_large):
        A[np.random.randint(0,n-1)][np.random.randint(0,n-1)] = max_val * n

    # plot s vs error
    error_scale = [i/10 for i in range(3,10)]
    s_scale = [s_calc(A,error_scale[i], n) for i in range(len(error_scale))]


    test_points = 50
    # each index - ((error, samples), success_rate)
    success_array = []

   

    trials = 20
    for p in range(test_points):
        error_p = np.random.uniform(error_scale[0], error_scale[-1])
        sample_p = int(np.random.uniform(s_scale[-1], s_scale[0]))
        # sample_p = int(np.random.uniform(2*n, n**2))
        num_success = 0
        print('New trial set:', p)
        for t in range(trials):
            print('trial:', t)
            A_til = sparsify(A, error_p, n, sample_p, True)
            num_success += sparse_error_success(A, A_til, error_p)
        success_array.append(((error_p, sample_p), num_success/trials))
    
    
    plt.plot(error_scale,s_scale)
    plt.xlabel('Error percent')
    plt.ylabel('Sample size')
    plt.title('Medium Variance Matrix - Large Sample Size')

    success_threshold = .5
    for val in range(test_points):
        if success_array[val][1] >= success_threshold:
            plt.plot(success_array[val][0][0], success_array[val][0][1], 'go')
        else:
            plt.plot(success_array[val][0][0], success_array[val][0][1], 'ro')

    
    plt.show()