# You should first set the path to your dataset and the output path in the config, and then run the following command based on your needs.
BMCNet:
EventNFS/nfs-syn/RGB-syn
python train.py -c config/train_EventZoom.yml
python train.py -c config/train_nfs.yml
python train.py -c config/train_RGB.yml


BMCNet_plain:
EventNFS/nfs-syn/RGB-syn
python train_plain.py -c config/train_EventZoom.yml
python train_plain.py -c config/train_nfs.yml
python train_plain.py -c config/train_RGB.yml
