#####################################
# run this script to add extra data #
# $1 is the path to the new data with
# the format below:
# 	$1:
#	    name_of_the_perosn:
#	    face0.jpg(png)
#	    face0.jpg(png)
#	    face0.jpg(png)
#           .
#           .
#           .
#####################################


cd src/align/
python3 align_dataset_mtcnn.py ../../$1 ../../data/lfw_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.5
cd ../../

# ./check.sh
