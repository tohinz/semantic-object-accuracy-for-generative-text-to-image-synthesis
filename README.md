# object-centric-adversarial-text-to-image-synthesis

# Calculate the SOA Scores (Semantic Object Accuracy)

How to calculate the SOA scores for a model:

1. TODO get the captions from SOA/captions and use your model to generate three images for each of the captions in the 80 files

2. Use your model to generate images from the specified captions

    1. each caption file contains the relevant captions for the given label, see [here](SOA/README.md) for the exact details
    2. create a new and empty folder 
    3. use each caption file to generate images for each caption and save the in a folder within the previously created empty folder, i.e. for each of the labels (0-79) there should be a new folder in the previously created folder and the folder structure should look like this
              * images
             * * label_00 -> folder contains images generated from captions for label 0
               * label_01 -> folder contains images generated from captions for label 0
               * ...
               * label_79 -> folder contains images generated from captions for label 0
    4. each new folder (that contains generated images) should contain the string "label_XX" somewhere in its name (make sure that integers are formated to two digits, e.g. "0", "02", ...) -> ideally give the folders the same name as the label files
    5. generate **three images for each caption** in each file
       * exception: for label "00" (person) randomly sample 30,000 captions and generate one image each for a total of 30,000 images
    6. in the end you should have 80 folders in the folder created in the step (2.2), each folder should have the string "label_XX" in it for identification, and each folder should contain the generated images for this label

3. Once you have generated images for each label you can calculate the SOA scores:
    1. TODO install requirements from SOA/requirements.txt
    

* 

* 
* * TODO install requirements from SOA/requirements.txt
  * TODO download the YOLO weights file and save it as ``SOA/yolov3.weights``
* * run ``python calculate_soa.py --images path/to/folder/created-in-first-step --output path/to/folder/where-results-are-saved --gpu 0``
* TODO if you also want to calculate IoU values:
* * 
* * run ``python calculate_soa.py --images path/to/folder/created-in-first-step --output path/to/folder/where-results-are-saved --gpu 0 --iou``
* calculating the SOA scores takes about 30-45 minutes (tested with a NVIDIA GTX 1080TI) depending on your hardware (not including the time it takes to generate the images)
* more detailed information [here](SOA/README.md)
