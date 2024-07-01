# TRANSFORMING THE BRAIN: A Hybrid CNN-Transformer Approach for Diagnosing Harmful Brain Activity

### Amit Erraguntla, Aviral Vaidya, and Sean Kim

## Introduction

An electroencephalogram (EEG) is a crucial tool for providing insights into the brain by using time series electrical activity at different nodes. Our goal was to create a model that can output a probability distribution for various classes of harmful brain activity: SZ (seizure), LPD (lateralized periodic discharge), GPD (generalized periodic discharge), LRDA (lateralized rhythmic delta activity), GRDA (generalized rhythmic delta activity), and OTHER. This is important because manual review of EEG recordings is time-intensive and expensive, and results can vary depending on the reviewer. We aim to improve EEG pattern classification accuracy and enable faster detection of harmful brain activity.

## Methodology & Preprocessing + Architecture

### Data Source

Our data is publicly available from Harvard Medical School (HMS), in the form of 50 second EEG and spectrogram data.

### Preprocessing

For each input, we have two spectrograms: 
- The spectrogram provided in the HMS dataset.
- A spectrogram we generate ourselves.

We use the raw EEG waveform time-series data and apply a wavelet + Fast Fourier transform before generating the spectrogram using the librosa library. Finally, we log transform and normalize both spectrograms before concatenating them together to pass into our model.

### Model Architecture

- **EfficientNetB0-Based CNN:** The preprocessed and concatenated input spectrograms pass through EfficientNetB0, a CNN architecture initialized with pre-trained weights from the ImageNet dataset. This creates 1024 channels for each input which contain important features of the spectrogram. This goes through a Global Average 2D Pooling layer which turns each channel into a single value.
- **Transformer Block:** The 1024-length sequence goes through a transformer that uses multi-headed attention to capture the relationships between the channels created by the CNN, with added dropout layers to prevent overfitting.
- **Final Output Layer:** The output is a probability distribution of the six categories: SZ, LPD, GPD, LRDA, GRDA, and OTHER. We use KL Divergence as our loss function to measure the difference between our predicted and actual probability distributions.

## Results

Below is a comparison of our model’s performance compared to others:

| Model                               | KL Divergence  |
| ----------------------------------- | -------------- |
| Fourier Transform + ViT             | 0.318432       |
| ViT Pre-Trained Vision Transformer  | 0.457420       |
| CNN + LSTM + Multimodal Input       | 0.500874       |
| **CNN + Transformer**                   | **0.550406**       |
| EEG + Spectrogram + 2D CNN          | 0.610243       |
| CNN + LSTM with only EEG data       | 1.101587       |
| 1D CNN (Time Series & Image)        | 1.106557       |

Our model achieves comparable performance to most other approaches, outperforming LSTM and CNN-based approaches, but underperforming against multimodal and vision transformer approaches.

## Challenges

While designing our project, we ran into several challenges:
- **Large Dataset and Complex Model Architecture:** Our dataset was large and our model architecture was very complex with many trainable parameters leading to long training times. To address this, we loaded data and trained the model in batches, saving the weights in between.
  
## Reflection

Overall, we felt like our project was a success. While we did not meet our stretch goal of improving performance compared to all other models on our task, our accuracy was comparable to most advanced approaches and actually outperformed some of them. Initially, we had planned to design a multimodal model for our task but ended up not doing so due to the unexpected complexity of our dataset. 

Our biggest takeaway from this project is that real datasets are HUGE. Compared to what we deal with in class, when people build models, they need to figure out ways of simply training the models such that their machine can handle it. We had to learn how to load and train our data in batches for this. In addition, data is far from perfect. Most datasets will be messy and hard to deal with, which is something we certainly experienced during this project. Preprocessing, cleaning, and augmentation can be a huge portion of the work.
