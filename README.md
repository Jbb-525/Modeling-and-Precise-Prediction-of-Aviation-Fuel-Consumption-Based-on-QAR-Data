# Modeling-and-Precise-Prediction-of-Aviation-Fuel-Consumption-Based-on-QAR-Data
This project aimed to replicate existing models from the literature and continuously optimize them by incorporating processed QAR data to predict aircraft fuel consumption.

## ① Model Replication

###  GA-BPNN [1]

  1. Implemented the Genetic Algorithm (GA) from scratch (Details in the GA-BPNN/GA.py).
  2. Implemented Levenberg-Marquardt second-order optimization (Details in the GA-BPNN/BPNN.py).
  3. Integrated GA with BPNN (Details in the GA-BPNN/GA-BPNN.ipynb).

### PSO-ELMAN [2]

  1. Implemented the PSO from scratch (Details in the PSO-Elman/PSO.py).
  2. Implemented the Elman from scratch (Details in the PSO-Elman/Elman.py).
  3. Integrated PSO with Elman (Details in the PSO-Elman/PSO-Elman.ipynb)

### CPCLS [3]

  1. Implemented a self-organizing constructive neural network (CNN) that features a cascade architecture (Details in the CPCLS/CPCLS.ipynb).

### CNN-LSTM [4]

  1. Implemented a dual-channel fusion CNN-LSTM model (Details in the CNN-LSTM/CNN-LSTM.ipynb).

## ② LSTM-Attention model created authors


## ③ Paper written based on the project are also presented

  "_Enhancing Aircraft Fuel Prediction with LSTM-Attention: Examining Lag Effects Across the Entire Flight_"

**Abstract:** This research is centered on achieving full-flight fuel consumption fitting using a single model, with a specific focus on emphasizing the potential lag effects in aircraft fuel consumption. The proposed LSTM-Attention model integrates the capabilities of the LSTM network to effectively extract correlation features and sequence features from the data. Simultaneously, the attention mechanism assumes a crucial role in accentuating temporal dependencies and lag effects associated with fuel consumption in distinct flight segments. The parameters of the model undergo meticulous optimization through adjustment experiments. Experimental results demonstrate that compared to traditional models like BPNN, ELMAN, and RNN, the proposed model more efficiently extracts fuel consumption features throughout the entire flight, reducing the average prediction error by 66.5% and improving stability by an average of 38.8%. This model not only contributes to advancing fuel-saving research based on Quick Access Recorder (QAR) data but also holds promise for fault diagnosis applications in aviation.
