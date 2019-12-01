from matplotlib import pyplot as plt
from matplotlib import image as img
from Sparse_alg import *
import sys


# Helper function that grayscales an image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Function that tests different values of error and s (number of samples)
# to create sparse matrix representations of A. The function then plots
# the success rate of the sparsification algorithm as a density graph.
def run_exp(A, trials, title, scale, plot_true_s):

	# Compute the dimension of A
    n = len(A)

    # Number of points: (scale - 1)^2
    error_scale = [i/scale for i in range(1,scale)]

    # true s plot
    s_scale = [s_calc(A,error_scale[i], n) for i in range(len(error_scale))]

    # values within true s
    # samples = np.linspace(s_scale[-1], s_scale[0], len(error_scale))

    # more resonable s values
    samples = np.linspace(n, 8*n**2, len(error_scale))

    # DEBUG: print the stable rank of A
    print(stable_rank(A))

    # each index - ((error, samples), success_rate)
    success_matrix = []
    p = 0

    for i in range(len(error_scale)):
        error_p = error_scale[i]
        for j in range(len(samples)):
            sample_p = int(samples[j])

            success_matrix.append([])
            num_success = 0
            # DEBUG print statement
            print('New trial set:', p)
            for t in range(trials):
                # DEBUG print statement
                print('trial:', t)
                A_til = sparsify(A, error_p, n, sample_p, True)
                num_success += sparse_error_success(A, A_til, error_p)
            success_matrix[i].append(((error_p, sample_p), num_success/trials))
            
            p+=1
        
    # true s plot
    if plot_true_s:
        plt.plot(error_scale,s_scale)

    plt.xlabel('Error percent')
    plt.ylabel('Sample size')
    plt.title(title)

    # Plot success rate of the sparsification as a function of error and s
    success_threshold = .5
    for i in range(len(error_scale)):
        for j in range(len(samples)):

            if success_matrix[i][j][1] >= success_threshold:
                plt.plot(success_matrix[i][j][0][0], success_matrix[i][j][0][1], 'go')
            else:
                plt.plot(success_matrix[i][j][0][0], success_matrix[i][j][0][1], 'ro')

    plt.show()

    # Function that tests accuracy of eigenvector/eigenvalue calculation with sparse matrix
def test_sparse_eig(A, error, n, true_s):
    A_til = []
    if (true_s):
        # sparse matrix with s samples from paper
        A_til = sparsify(A, error, n, s_calc(A, error, n), True)
    else:
        # sparse matrix with O(n^2) samples
        A_til = sparsify(A, error, n, n**2, True)

    # Calculate eigenvector and eigenvalue pair
    eig_vec, eig_val = np.linalg.eig(A)
    eig_vec_A_til, eig_val_A_til = np.linalg.eig(A_til)

    # Print error
    print(np.linalg.norm(eig_vec - eig_vec_A_til,2)/np.linalg.norm(eig_vec))

# shows density of matrices
def display_matrix(A, title):
    plt.matshow(A, cmap=plt.cm.Greys)
    plt.title(title)
    plt.show()

if __name__ == '__main__':

    img_name = sys.argv[1]
    trials = int(sys.argv[2])
    scale = int(sys.argv[3])
    plot_true_s = bool(sys.argv[4])

    np.random.seed(1234)

    # init matricies from images
    img_rgb = img.imread(img_name)
    img_gray = rgb2gray(img_rgb)
    # plt.imshow(img_gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    # plt.show()

    # guarantee testing
    # trials = 3
    # scale = 20
    # plot_true_s = True
    run_exp(img_gray, trials, img_name+' Image', scale, plot_true_s)

    # matrix display
    display_matrix(img_gray, 'Original Image')
    for e in err:
        s = s_calc(img_gray, e, n)
        display_matrix(sparsify(img_gray, e, n, s, True), 'Error: '+str(e))

        # No truncation
        # display_matrix(sparsify(img_gray, e, n, s, False))

    # Eigenvector computation test
    # err = [i/5 for i in range(1,5)]
    # n = 50

    # print('Error:', err)
    # print('Eigenvector computation test with true s samples')
    # test_sparse_eig(img_gray, err, n, True)

    # print('Eigenvector computation test with O(n^2) samples')
    # test_sparse_eig(img_gray, err, n, False)