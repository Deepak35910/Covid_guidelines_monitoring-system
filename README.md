# Covid_guidelines_monitoring-system

# PROBLEM STATEMENT –
As there is no vaccine available for Corona virus yet, the best way to control the spread is  by maintaining social distance and wearing mask. This project will use live CCTV footage to detect social distancing and will use audio device to instruct people to maintain social distancing and to put on mask. Also, it will detect coughing and sneezing from CCTV footage.


for social distancing run file - People_tracker_yolo.py


for mask detection run file - yolo-3-video.py


for coughing sneezing

People_tracker.py - run this to get encoding of a person action

sneezecolab.py - to create data (X and Y) 

cough_sneeze.ipynb - modelling

sneeze.py - checking a video and saving the video
(sneezecolab.py and cough_sneeze.ipynb  was written and run in colab)


for color detection refer colour.ipynb



# Model performance

for colour - 93% accuracy

for cough-sneezing - 95% accuracy

for mask - 99% accuracy



# Dataset - 

for colour - we scrapped around 5000 images for 13 different color

for cough-sneezing - https://web.bii.a-star.edu.sg/~chengli/FluRecognition.htm

for mask - https://github.com/prajnasb/observations/tree/master/experiements/data
(we got more data from different sources)



# reference - 

eMaster Class Academy (https://www.youtube.com/channel/UCtfTf1nNJQ4PbUDqj-Q48rw)

Adrian Rosebrock (https://www.pyimagesearch.com/)



# Team member

Deepak Yadav

Parag Bhagat

Ikpreet singh
