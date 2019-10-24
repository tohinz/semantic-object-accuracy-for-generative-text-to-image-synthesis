#   Semantic Object Accuracy for Generative Text-to-Image Synthesis
Code for our paper [  Semantic Object Accuracy for Generative Text-to-Image Synthesis](https://www.google.de).

Contents:
* [Calculate SOA Scores](#calculate-the-soa-scores-semantic-object-accuracy)

# Calculate the SOA Scores (Semantic Object Accuracy)

How to calculate the SOA scores for a model:

1. The captions are in ``SOA/captions``
    1. each file is named ``label_XX_XX.pkl`` describing for which labels the captions in the file are
    2. load the file with pickle
        * ```python
             import pickle 
             with open(label_XX_XX.pkl, "rb") as f:
                 captions = pickle.load(f)
          ```
    3. each file is a list and each entry in the list is a dictionary containing information about the caption:
        * ```python
            [{'image_id': XX, 'id': XX, 'idx': [XX, XX], 'caption': u'XX'}, ...]
          ```
        * where ``'idx': [XX, XX]`` gives the indices for the validation captions in the commonly used captions file from [AttnGAN](https://github.com/taoxugit/AttnGAN)
2. Use your model to generate images from the specified captions

    1. each caption file contains the relevant captions for the given label
    2. create a new and empty folder 
    3. use each caption file to generate images for each caption and save the images in a folder within the previously created empty folder, i.e. for each of the labels (0-79) there should be a new folder in the previously created folder and the folder structure should look like this
        * images
            * label_00 -> folder contains images generated from captions for label 0
            * label_01 -> folder contains images generated from captions for label 1
            * ...
            * label_79 -> folder contains images generated from captions for label 79
    4. each new folder (that contains generated images) should contain the string "label_XX" somewhere in its name (make sure that integers are formated to two digits, e.g. "0", "02", ...) -> ideally give the folders the same name as the label files
    5. generate **three images for each caption** in each file
        * exception: for label "00" (person) randomly sample 30,000 captions and generate one image each for a total of 30,000 images
    6. in the end you should have 80 folders in the folder created in the step (2.ii), each folder should have the string "label_XX" in it for identification, and each folder should contain the generated images for this label

3. Once you have generated images for each label you can calculate the SOA scores:
    1. Install requirements from ``SOA/requirements.txt`` (we use Python 3.5.2)
    2. TODO download the YOLOv3 weights file and save it as ``SOA/yolov3.weights``
    3. run ``python calculate_soa.py --images path/to/folder/created-in-step-2ii --output path/to/folder/where-results-are-saved --gpu 0``

4. If you also want to calculate IoU values check the detailed instructions [here](SOA/README.md)
5. Calculating the SOA scores takes about 30-45 minutes (tested with a NVIDIA GTX 1080TI) depending on your hardware (not including the time it takes to generate the images)
6. More detailed information (if needed) [here](SOA/README.md)

