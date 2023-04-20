# Face_emotion_detect
A model that can detect face emotion like negative ,positive and normal based on Face-emotion-yolov5 with the link https://github.com/SalehAliAlyahri/Face-emotion-yolov5
I use the model in Face-emotion-yolov5 project for transfering learning
Based on transfer learning, I used three labels: normal, negative, and positive. I believe that human emotions are not just sad and happy, sometimes there may also be anger, and these emotions are all negative. Therefore, I replaced happy with positive and sad with negative.
From the results, the recognition accuracy of normal is the highest, but sometimes positive and negative cannot be distinguished. I think this is due to the small size of the dataset and the similarity of the photos of positive and negative emotions, causing overfitting.
I still change the detect.py in yolov5 so that it can be  packaged into an exe file better
# note
the accuracy is not ideal,you can still use this model in this project for transfering learning to achieve your purpose you want.
