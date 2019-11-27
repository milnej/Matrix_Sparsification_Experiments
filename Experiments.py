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
def run_exp(A, trials, title):

	# Compute the dimension of A
    n = len(A)

    # 49 total points
    error_scale = [i/10 for i in range(3,10)]
    #289 total points
    # error_scale = [i/20 for i in range(3,20)]

    # true s plot
    s_scale = [s_calc(A,error_scale[i], n) for i in range(len(error_scale))]

    # values within true s
    # samples = np.linspace(s_scale[-1], s_scale[0], len(error_scale))

    # more resonable s values
    samples = np.linspace(n, n**2, len(error_scale))

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
    # plt.plot(error_scale,s_scale)

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

if __name__ == '__main__':

    img_name = sys.argv[1]

    #np.random.seed(1234)

    # init matricies from images
    img_rgb = img.imread(img_name+'.png')
    img_gray = rgb2gray(img_rgb)
    # plt.imshow(img_gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    # plt.show()

    trials = 1

    run_exp(img_gray, trials, 'Beach Image')
    

