# store the aligned face to data/lfw_160 (image size: 160*160*3)
cd src/align/
python3 align_dataset_mtcnn.py ../../data/lfw-deepfunneled/ ../../data/lfw_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.5
cd ../../
