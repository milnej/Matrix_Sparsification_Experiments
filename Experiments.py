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
def run_exp(A, test_points, trials, title):

	# Compute the dimension of A
    n = len(A)

    # Set up scale for error vs s plot
    error_scale = [i/10 for i in range(3,10)]
    s_scale = [s_calc(A,error_scale[i], n) for i in range(len(error_scale))]

    # DEBUG: print the stable rank of A
    print(stable_rank(A))

    # Create an array that holds the results of each (error, s) pair
    # Each element in the array - ((error, s), success_rate)
    success_array = []

    # Start testing "test_points" number of points
    for p in range(test_points):
    	# Generate values for the (error, s) pairs
        error_p = np.random.uniform(error_scale[0], error_scale[-1])
        sample_p = int(np.random.uniform(s_scale[-1], s_scale[0]))

        # samples a low amount of values s (2n - n^2)
        # sample_p = int(np.random.uniform(2*n, n**2))

        # Calculate the success rate for each (error, s) pair
        num_success = 0
        # DEBUG print statement
        print('New trial set:', p, "error:", error_p, "s:", sample_p)
        for t in range(trials):
        	# DEBUG print statement
            print('trial:', t)
            A_til = sparsify(A, error_p, n, sample_p, True)
            num_success += sparse_error_success(A, A_til, error_p)
        success_array.append(((error_p, sample_p), num_success/trials))
    
    # Plot results of the experiments
    plt.plot(error_scale,s_scale)
    plt.xlabel('Error percent')
    plt.ylabel('Sample size')
    plt.title(title)

    # Plot success rate of the sparsification as a function of error and s
    success_threshold = .5
    # Success: success rate is above the threshold
    # Failure: success rate is below the threshold
    for val in range(test_points):
        if success_array[val][1] >= success_threshold:
            plt.plot(success_array[val][0][0], success_array[val][0][1], 'go')
        else:
            plt.plot(success_array[val][0][0], success_array[val][0][1], 'ro')

    plt.show()

if __name__ == '__main__':

    img_name = sys.argv[1]

    #np.random.seed(1234)

    # init matricies from images
    img_rgb = img.imread(img_name+'.png')
    img_gray = rgb2gray(img_rgb)
    # plt.imshow(beach_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    # plt.show()

    test_points = 20
    trials = 1

    # First experiment: Beach Image
    run_exp(img_gray, test_points, trials, 'Beach Image')
    

