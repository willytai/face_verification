######################################################################
# Preperation:                                                       #
# Download the LFW database and place them in data/                  #
# Download the facenet model and place the whole directory in model/ #
######################################################################

# store the aligned face to data/lfw_160 (image size: 160*160*3)
cd src/align/
python3 align_dataset_mtcnn.py ../../data/lfw-deepfunneled/ ../../data/lfw_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.5
cd ../..


# run facenet to get the embedding vector, store the vectors to data/embeddings
cd src/
python3 to_face_embedding.py ../data/lfw_160/ ../model/20180408-102900/
cd ../

# validat on LFW (10-fold)
cd src/
python3 validate.py cosine > result.txt
cd ../
