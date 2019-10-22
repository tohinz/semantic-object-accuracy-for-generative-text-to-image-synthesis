# Details for calculating the SOA Scores (Semantic Object Accuracy)

## Work with the caption files
To load the captions: load a caption file, get the captions, and generate images:
```python
import pickle
import my_model

# load the caption file
with open(label_01_bicycle.pkl, "rb") as f:
    captions = pickle.load(f)

# iterate over the captions and generate three images each
for caption in captions:
    current_caption = caption["caption"]
    for idx in range(3):
        my_generated_image = my_model(current_caption)
        save("images/label_01_bicycle/my_generated_image.png)    
```

Alternatively, if you're working with the ``captions.pickle`` file from the [AttnGAN](https://github.com/taoxugit/AttnGAN) and their dataloader you can use the provided ``idx`` to load the file directly from the file:
```python
import pickle
import my_model

# load the AttnGAN captions file
with open(captions.pickle, "rb") as f:
    attngan_captions = pickle.load(f)
test_captions = attngan_captions[1]

# load the caption file
with open(label_01_bicycle.pkl, "rb") as f:
    captions = pickle.load(f)

# iterate over the captions and generate three images each
for caption in captions:
    current_caption_idx = caption["idx"]
    # new_ix is the index for the filenames
    new_ix = [current_caption_idx[0]]
    # new_sent_ix is the index to the exact caption, e.g. use it for 
    # caps, cap_len = get_caption(new_sent_ix)
    new_sent_ix = [current_caption_idx[0]*5+current_caption_idx[i][1]]
    for idx in range(3):
        ...
```

For the file ``label_00_person.pkl`` we randomly sample 30,000 captions and generate one image each:
```python
import pickle
import random
import my_model

# load the caption file
with open(label_00_person.pkl, "rb") as f:
    captions = pickle.load(f)

caption_subset = random.sample(captions, 30000)

# iterate over the captions and generate three images each
for caption in caption_subset:
    current_caption = caption["caption"]
    my_generated_image = my_model(current_caption)
    save("images/label_00_person/my_generated_image.png)    
```

## Calculating IoU Scores
For the IoU scores it is important that you use the same label mappings as we. Our labels are the following (also found in ``data/coco.names``) as per standard:
0       person
1       bicycle
2       car
3       motorbike
4       aeroplane
5       bus
6       train
7       truck
8       boat
9       traffic light
10      fire hydrant
11      stop sign
12      parking meter
13      bench
14      bird
15      cat
16      dog
17      horse
18      sheep
19      cow
20      elephant
21      bear
22      zebra
23      giraffe
24      backpack
25      umbrella
26      handbag
27      tie
28      suitcase
29      frisbee
30      skis
31      snowboard
32      sports ball
33      kite
34      baseball bat
35      baseball glove
36      skateboard
37      surfboard
38      tennis racket
39      bottle
40      wine glass
41      cup
42      fork
43      knife
44      spoon
45      bowl
46      banana
47      apple
48      sandwich
49      orange
50      broccoli
51      carrot
52      hot dog
53      pizza
54      donut
55      cake
56      chair
57      sofa
58      pottedplant
59      bed
60      diningtable
61      toilet
62      tvmonitor
63      laptop
64      mouse
65      remote
66      keyboard
67      cell phone
68      microwave
69      oven
70      toaster
71      sink
72      refrigerator
73      book
74      clock
75      vase
76      scissors
77      teddy bear
78      hair drier
79      toothbrush


class labels -> make sure they're right



how to load and process a caption file



instructions about calculating iou values

    1. TODO save dicts
    2. TODO make sure you use the same labeling as we do
    3. run ``python calculate_soa.py --images path/to/folder/created-in-first-step --output path/to/folder/where-results-are-saved --gpu 0 --iou``
