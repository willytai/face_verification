##################
# $1 query image #
##################


# clean

rm -rf query

# create the aligned auery imgae

mkdir tmp; cd tmp;
mkdir query; cd ../;
cp $1 tmp/query/

cd src/align
python3 align_dataset_mtcnn.py ../../tmp ../../ --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.5
cd ../../

rm -rf *txt
rm -rf tmp




# get the face embedding of the image

cd src/
python3 to_face_embedding.py ../data/lfw_160/ ../model/20180408-102900/ --query_dir ../query/
cd ../




# search for the best match

cd src/
python3 search.py
cd ../
