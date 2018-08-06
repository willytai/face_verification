import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine
from scipy.misc import imread




def visualize(result, query_path, Name):

	pic = [os.path.join(result, p) for p in os.listdir(result) if p.endswith('png') or p.endswith('jpg')]

	if len(pic) > 10:

		pic = pic[:10]

	images = []

	for p in pic:

		images.append(imread(p))

	query = [os.path.join(query_path, p) for p in os.listdir(query_path) if p.endswith('png') or p.endswith('jpg')]

	assert len(query) == 1

	query = query[0]

	show_images(images, imread(query), Name)

def show_images(images, query, Name, cols = 3, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = [Name]*len(images)
    fig = plt.figure()
    a = fig.add_subplot(cols, 5, 3)
    plt.imshow(query)
    a.set_title('query image')
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, 5, n + 6)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    plt.savefig('../query/'+Name+'.jpeg')

def cal_ave_dist(query, choices):

	dist = []

	for c in choices:

		dist.append(cosine(query, np.load(c)))

	return np.array(dist).mean()

def get_min_dist(query, choices):

	dist = 100

	for c in choices:

		new_dist = cosine(query, np.load(c))

		# speed up
		if new_dist > 1:

			return new_dist

		if dist > new_dist:

			dist = new_dist

	return dist

def print_format(t):

	score = t[1]
	score = 1 - score
	
	print ('{:29}  {:.2f}%'.format(t[0], score*100))

def main():
	query_path = '../query/'
	data_path  = '../data/lfw_160/'

	names = os.listdir(data_path)
	q     = [f for f in os.listdir(query_path) if f.endswith('npy')]

	assert len(q) == 1, 'expect 1 query img, %d found instead.' % len(q)

	query = np.load(os.path.join(query_path, q[0]))

	top_10 = []

	start_time = time.time()

	for pp, N in enumerate(names):

		ss = time.time()
		print ('Checking %s...' % N, end='')

		try:
			pic = os.listdir(os.path.join(data_path, N))
		except:
			continue

		pic = [os.path.join(data_path, N, p) for p in pic if p.endswith('npy')]

		# dist = cal_ave_dist(query, pic)
		dist = get_min_dist(query, pic)

		if len(top_10) < 10:

			top_10.append((N, dist))
			top_10 = sorted(top_10, key=lambda x : x[1])

		elif dist < top_10[-1][1]:
			
			top_10[-1] = (N, dist)
			top_10 = sorted(top_10, key=lambda x : x[1])

		# else:

		# 	break

		# break if found a result with cosine distance less than 0.05
		if top_10[0][1] <= 0.1:
			break

		print ('{:.2f}s'.format(time.time()-ss))


	print ('')
	print ('='*7, 'Result (top-10-match)', '='*7)
	for t in top_10:
		print_format(t)
	print ('Took {:.2f} seconds (searched over {:5d} people)'.format(time.time()-start_time, pp+1))


	result = os.path.join(data_path, top_10[0][0])

	visualize(result, query_path, top_10[0][0])

if __name__ == '__main__':
	main()