# object-centric-adversarial-text-to-image-synthesis

# Calculate the SOA (Semantic Object Accuracy Score)

Requirements:

* clone this Git
* TODO get the captions from SOA/captions and use your model to generate three images for each of the captions in the 80 files
* * generated images should be in a nested folder
  * create a new folder and empty folder
  * for each of the labels (0-79) there should be a new folder in the previously created folder
  * each new folder should contain the string "label_XX" somewhere in its name (make sure that integers are formated to two digits, e.g. "0", "02", ...) -> ideally give the folders the same name as the label files
  * use each caption file to generate three images for each caption in the file and save them in the corresponding folder
  * * exception: for label "00" (person) sample 30,000 captions and generate one image each for a total of 30,000 images
  * in the end you should have 80 folders in the folder created in the first step, each folder should have the string "label_XX" in it for identification, and each folder should contain the generated images for this label
* TODO once you have generated all images:
* * TODO install requirements from SOA/requirements.txt
* * TODO ``python calculate_soa.py --images path/to/folder/created-in-first-step --output path/to/folder/where-results-are-saved --gpu 0``
* TODO if you also want to calculate IoU values:
* * 
* * TODO ``python calculate_soa.py --images path/to/folder/created-in-first-step --output path/to/folder/where-results-are-saved --gpu 0 --iou``
* calculating the SOA scores takes about 30-60 minutes (tested with a NVIDIA GTX 1080TI) depending on your hardware (not including the time it takes to generate the images)