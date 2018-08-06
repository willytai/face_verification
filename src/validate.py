import numpy as np
import sys, os
import argparse

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from scipy.spatial.distance import cosine, euclidean, cityblock

def save_embeddings_to_data():

	embeddings = np.load('../data/embeddings.npy')

	embed_path = []

	skip = True
	with open('../data/pairs.txt', 'r') as f:
		for line in f:
			if skip:
				skip = False
				continue

			line = line.strip().split('\t')

			if len(line) == 3:

				embed_path.append('../data/lfw_160/{}/{}_%04d.npy'.format(line[0], line[0]) % int(line[1]))
				embed_path.append('../data/lfw_160/{}/{}_%04d.npy'.format(line[0], line[0]) % int(line[2]))

			elif len(line) == 4:

				embed_path.append('../data/lfw_160/{}/{}_%04d.npy'.format(line[0], line[0]) % int(line[1]))
				embed_path.append('../data/lfw_160/{}/{}_%04d.npy'.format(line[2], line[2]) % int(line[3]))

	assert len(embed_path) == embeddings.shape[0]

	for vec, path in zip(embeddings, embed_path):

		# abort of files already exist
		if os.path.isfile(path):
			break

		print (path)
		np.save(path, vec)

def preprocess(data, dist='cosine'):

	train_x = []
	train_y = []

	for d in data:

		if dist == 'cosine':

			train_x.append(cosine(d[0], d[1]))

		elif dist == 'euclidean':

			train_x.append(euclidean(d[0], d[1]))

		elif dist == 'cityblock':

			train_x.append(cityblock(d[0], d[1]))

		train_y.append(d[2])

	return np.array(train_x).reshape((len(train_x), 1)), np.array(train_y)

def main(args):

	# save_embeddings_to_data()

	data = []
	result = []

	# read testing pairs
	skip = True
	with open('../data/pairs.txt', 'r') as f:
		for line in f:
			if skip:
				skip = False
				continue

			line = line.strip().split('\t')

			if len(line) == 3:

				embed_path_1 = ('../data/lfw_160/{}/{}_%04d.npy'.format(line[0], line[0]) % int(line[1]))
				embed_path_2 = ('../data/lfw_160/{}/{}_%04d.npy'.format(line[0], line[0]) % int(line[2]))
				data.append((np.load(embed_path_1), np.load(embed_path_2), 1))

			elif len(line) == 4:

				embed_path_1 = ('../data/lfw_160/{}/{}_%04d.npy'.format(line[0], line[0]) % int(line[1]))
				embed_path_2 = ('../data/lfw_160/{}/{}_%04d.npy'.format(line[2], line[2]) % int(line[3]))
				data.append((np.load(embed_path_1), np.load(embed_path_2), 0))

			else:
				assert False

		data = np.array(data)

	# 10-fold cross validation
	kf = KFold(n_splits=10)

	for i, (train_idx, test_idx) in enumerate(kf.split(data)):

		print ('\rTraining SVM...(Fold {:2})'.format(i+1), end='', flush=True)

		train_x, train_y = preprocess(data[train_idx], dist=args.distance_metric)
		test_x,  test_y  = preprocess(data[test_idx],  dist=args.distance_metric)

		clf = SVC(C=args.C, kernel=args.kernel)

		clf.fit(train_x, train_y)

		train_score = clf.score(train_x, train_y)
		valid_score = clf.score(test_x,  test_y)

		result.append((train_score, valid_score))


	ave = []
	print ('')
	print ('-'*116)
	print ('|              | {} |'.format(" | ".join(['Fold %2d' % i for i in np.arange(1, 11)])))
	print ('-'*116)

	print ('| Training Acc |', end='')
	for fold in result:
		print (' {:.5f} |'.format(fold[0]), end='')
	print ('')
	print ('-'*116)

	print ('| Testing Acc  |', end='')
	for fold in result:
		print (' {:.5f} |'.format(fold[1]), end='')
		ave.append(fold[1])
	print ('')
	print ('-'*116)

	print ('Distance Metric: {}, SVM kernel: {}, SVM penalty parameter: {}'.format(args.distance_metric, args.kernel, args.C))
	print ('Average Acc on LFW: {:.5f}'.format(np.array(ave).mean()))

def parse_arg(argv):

	parser = argparse.ArgumentParser()

	parser.add_argument('distance_metric', type=str,
		help='Distance Metric:	cosine; euclidean')

	parser.add_argument('--C', type=float,
		help='Penalty Parameter', default=1.0)

	parser.add_argument('--kernel', type=str,
		help='SVM kernel', default='linear')

	return parser.parse_args(argv)





if __name__ == '__main__':
	main(parse_arg(sys.argv[1:]))
