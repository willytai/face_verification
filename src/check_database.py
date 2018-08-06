import numpy as np
import argparse
import os
import sys
import re


def main(args):

	directory = os.listdir(args.data_path)

	check_list = list()

	for dirs in directory:

		try:
			files = os.listdir(os.path.join(args.data_path, dirs))
		except:
			# skip those that are not directories
			continue

		for file in files:

			temp = re.sub(r".jpg", ".npy", file)
			temp = re.sub(r".png", ".npy", temp)
			path = os.path.join(args.data_path, dirs, temp)

			if not os.path.isfile(path):

				check_list.append(os.path.join(dirs, file))

	print ('{} faces waiting to be processed'.format(len(check_list)))

	with open(os.path.join(args.data_path, 'check_list.txt'), 'w+') as f:

		f.write('{}\n'.format(len(check_list)))

		for file in check_list:

			f.write('{}\n'.format(file))


	# paths , actual_issame = get_paths(args.data_path, os.path.join(args.data_path, 'check_list.txt'), np.array([[]]), list())

	# print (paths[-10:])


def parse_arg(argv):

	parser = argparse.ArgumentParser()

	parser.add_argument('data_path', type=str,
		help='path to database')

	return parser.parse_args(argv)



def get_paths(database_path, check_list_path, paths, actual_issame):

	paths = list(paths.reshape(-1))

	with open(check_list_path, 'r') as f:

		for line in f.readlines()[1:]:

			paths.append(os.path.join(database_path, line.strip()))

	# make the number of paths even
	if len(paths) % 2 == 1:

		paths.append(paths[-1])

	for i in range(len(paths)//2):

		actual_issame.append(True)

	return np.expand_dims(np.array(paths), 1), actual_issame

def get_batch_size(default, img_num):

	if img_num % default == 0:

		return default

	size = [1]

	for i in range(2, 181):

		if img_num % i == 0:

			size.append(i)

	return size[-1]

def save_embed(paths, embeddings):

	assert paths.shape[0] == embeddings.shape[0]

	paths = paths.reshape(-1)

	for i, pth in enumerate(paths):

		pth = re.sub(r".jpg", ".npy", pth)
		pth = re.sub(r".png", ".npy", pth)

		print ('saveing %s' % pth)

		np.save(pth, embeddings[i])

def get_query_path(dir_path, paths, actual_issame):

	assert paths.shape[0] == 0, 'database not configured, run check.sh first before querying images'

	files = os.listdir(dir_path)

	assert len(files) == 1, 'to many query image! expect 1, got {} instead.'.format(len(files))

	paths = list(paths.reshape(-1))

	paths.append(os.path.join(dir_path, files[0]))
	paths.append(paths[-1])

	return np.expand_dims(np.array(paths), 1), [True]


if __name__ == '__main__':
	main(parse_arg(sys.argv[1:]))