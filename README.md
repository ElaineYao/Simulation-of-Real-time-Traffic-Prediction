# Simulation-of-Real-time-Traffic-Prediction
This is a real-time version of [Trip Duration Prediction Using NN](https://github.com/ElaineYao/Trip-duration-prediction-using-NN).
Dataset can also be found in the link above.

# How to run
- 'python file_reader.py | python preprocess.py | python train.py'

# Dataflow
It uses UNIX pipe to send the output(JSON string) of one process to another process for further processing. 

![avatar](https://drive.google.com/file/d/1SdFAiihVURVQzUucRLduLARumoI0ZMBD/view?usp=sharing)
