from matplotlib import pyplot as plt
from matplotlib import image as img
from Sparse_alg import *
import sys

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def run_exp(A, test_points, trials, title):

    n = len(A)

    # plot s vs error
    error_scale = [i/10 for i in range(3,10)]
    s_scale = [s_calc(A,error_scale[i], n) for i in range(len(error_scale))]

    print(stable_rank(A))

    # each index - ((error, samples), success_rate)
    success_array = []

    for p in range(test_points):
        error_p = np.random.uniform(error_scale[0], error_scale[-1])
        
        # samples a high amount of values s (based on the paper)
        sample_p = int(np.random.uniform(s_scale[-1], s_scale[0]))

        # samples a low amount of values s (2n - n^2)
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
    plt.title(title)

    success_threshold = .5
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

    run_exp(img_gray, test_points, trials, 'Beach Image')
    

