# Tensorflow Image Extractor

<p align="center"><br><img src="images/Main.svg" alt="drawing" height="450"/></p>

You need to install **TensorFlow** from [here](https://www.tensorflow.org/install)

WASSCL R-CNN is validated on the PASCAL VOC 2007 benchmark, and obtains 90% of a fully- supervised Faster R-CNN's performance (measured using mAP) with only 15% of fully-supervised annotations together with image-level annotations for the rest images. The problem of object localization is the most difficult part of object detection. One approach is that we use sliding window of different size to locate objects in the image. Convolutional Neural Network. CNN is a supervised type of Deep learning, most preferable used in image recognition and computer vision.

Suppose you are working with MNIST dataset, you know each image in MNIST is 28 x 28 x 1(black & white image contains only 1 channel). Total number of neurons in input layer will 28 x 28 = 784, this can be manageable. What if the size of image is 1000 x 1000 which means you need 10⁶ neurons in input layer. Oh! This seems a huge number of neurons are required for operation. It is computationally ineffective right. So here comes Convolutional Neural Network or CNN. In simple word what CNN does is, it extract the feature of image and convert it into lower dimension without loosing its characteristics. In the following example you can see that initial the size of the image is 224 x 224 x 3. If you proceed without convolution then you need 224 x 224 x 3 = 100, 352 numbers of neurons in input layer but after applying convolution you input tensor dimension is reduced to 1 x 1 x 1000. It means you only need 1000 neurons in first layer of feedforward neural network.

Reference Code :
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
Mask R-CNN uses anchor boxes to detect multiple objects, objects of different scales, and overlapping objects in an image. This improves the speed and efficiency for object detection. Anchor boxes are a set of predefined bounding boxes of a certain height and width. Faster R-CNN is a single-stage model that is trained end-to-end. It uses a novel region proposal network (RPN) for generating region proposals, which save time compared to traditional algorithms like Selective Search. It uses the ROI Pooling layer to extract a fixed-length feature vector from each region proposal. Convolutional neural network is composed of multiple building blocks, such as convolution layers, pooling layers, and fully connected layers, and is designed to automatically and adaptively learn spatial hierarchies of features through a backpropagation algorithm.

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
	
This is the README file for Tensorflow Image Extractor repository. [^1]

[^1]: By : Vivek Padia.
