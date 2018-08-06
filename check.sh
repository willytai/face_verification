###########################################
# run this script after updating database #
###########################################

cd src/
python3 check_database.py ../data/lfw_160/
python3 to_face_embedding.py ../data/lfw_160/ ../model/20180408-102900/ --check_list ../data/lfw_160/check_list.txt
cd ..
