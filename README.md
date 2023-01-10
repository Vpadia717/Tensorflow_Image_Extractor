# Tensorflow Image Extractor

Here we have made a Village helping application using java libraries like **Lottie** or **_TensorFlow Lite_**.

<p align="center"><br><img src="app/src/main/res/drawable/villageera.svg" alt="drawing" height="650"/></p>

It is a *Villager supporting app* where the user will get **_A right for making post regarding the complaints in the village also they can suggest or advice_**.

The *Sarpanch* can view the posts **_also he can make a notice for whole village with in his finger tips_**.

1. You need to install TensorFlow Lite from [here](https://www.tensorflow.org/)

```Python
#@title Image Selection (don't forget to execute the cell!) { display-mode: "form"}
selected_image = 'Beach' # @param ['Beach', 'Dogs', 'Naxos Taverna', 'Beatles', 'Phones', 'Birds']
flip_image_horizontally = False #@param {type:"boolean"}
convert_image_to_grayscale = False #@param {type:"boolean"}

image_path = IMAGES_FOR_TEST[selected_image]
image_np = load_image_into_numpy_array(image_path)

# Flip horizontally
if(flip_image_horizontally):
  image_np[0] = np.fliplr(image_np[0]).copy()

# Convert image to grayscale
if(convert_image_to_grayscale):
  image_np[0] = np.tile(
    np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

plt.figure(figsize=(24,32))
plt.imshow(image_np[0])
plt.show()
```

### Important Instructions :

* The *Admin* or *Sarpanch* needs to login to utilize the services.

Reference Code : 
```Python
label_id_offset = 0
image_np_with_detections = image_np.copy()

# Use keypoints if available in detections
keypoints, keypoint_scores = None, None
if 'detection_keypoints' in result:
  keypoints = result['detection_keypoints'][0]
  keypoint_scores = result['detection_keypoint_scores'][0]

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections[0],
      result['detection_boxes'][0],
      (result['detection_classes'][0] + label_id_offset).astype(int),
      result['detection_scores'][0],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.30,
      agnostic_mode=False,
      keypoints=keypoints,
      keypoint_scores=keypoint_scores,
      keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

plt.figure(figsize=(24,32))
plt.imshow(image_np_with_detections[0])
plt.show()
```
Reference Images : <br>
	<br><img src="app/src/main/res/drawable/Splash.svg" alt="drawing" height="450"/>
	<img src="app/src/main/res/drawable/Login.svg" alt="drawing" height="450"/>
	<img src="app/src/main/res/drawable/SignUp.svg" alt="drawing" height="450"/>
	<br><img src="app/src/main/res/drawable/Dashboard.svg" alt="drawing" height="450"/>
	<img src="app/src/main/res/drawable/Post Creation.svg" alt="drawing" height="450"/>
	<img src="app/src/main/res/drawable/View Post.svg" alt="drawing" height="450"/>
	
This is the README file for Village Era repository. [^1]

[^1]: By : Village Era.
