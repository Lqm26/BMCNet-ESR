# dataset preparation

## real-world data
First download real data from the [site](https://sites.google.com/view/EventZoom), then run `convert_eventzoom.py`.

## synthetic data
NFS-syn:
First download NFS data from the [site](http://ci2cv.net/nfs/index.html), then using the DVS-Voltmeter [site](https://github.com/Lynn0306/DVS-Voltmeter) to generate simulated events (To obtain images of different scales using cv2.resize() and generate events respectively). Finally run `syn_nfs.py`.

RGB-syn:
First download RGB-DAVIS data from the [site](https://sites.google.com/view/guided-event-filtering), then using the DVS-Voltmeter [site](https://github.com/Lynn0306/DVS-Voltmeter) to generate simulated events (To obtain images of different scales using cv2.resize() and generate events respectively). Finally run `syn_RGB.py`.
