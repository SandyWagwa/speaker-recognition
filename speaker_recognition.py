# speaker_recognition.py

import os
import numpy as np
import librosa
from pydub import AudioSegment
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Function to extract MFCC features from an audio file
def mfcc_extraction(audio_filename, hop_duration, num_mfcc):
    """
    Extract MFCC features from an audio file.

    Parameters:
    - audio_filename: str, path to the .wav audio file
    - hop_duration: float, hop length in seconds (e.g., 0.015s)
    - num_mfcc: int, number of MFCC features to extract

    Returns:
    - mfcc.T: numpy array, MFCC features transposed
    """
    speech = AudioSegment.from_wav(audio_filename)
    samples = np.array(speech.get_array_of_samples(), dtype=np.float32)
    sampling_rate = speech.frame_rate

    # Normalize samples
    samples = samples / np.max(np.abs(samples))

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=sampling_rate,
        hop_length=int(sampling_rate * hop_duration),
        n_mfcc=num_mfcc
    )

    return mfcc.T

# Function to train a GMM model
def learningGMM(features, n_components, max_iter):
    """
    Train a GMM model on the provided features.

    Parameters:
    - features: numpy array, MFCC features
    - n_components: int, number of mixture components
    - max_iter: int, maximum number of iterations

    Returns:
    - gmm: GaussianMixture object, trained GMM model
    """
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter)
    gmm.fit(features)
    return gmm

# Speaker recognition function
def speaker_recognition(audio_file_name, gmms):
    """
    Recognize the speaker of the given audio file using the provided GMMs.

    Parameters:
    - audio_file_name: str, path to the test audio file
    - gmms: list of GaussianMixture objects, trained GMMs for each speaker

    Returns:
    - speaker_id: int, index of the recognized speaker in the gmms list
    """
    mfcc_test = mfcc_extraction(audio_file_name, hop_duration=0.015, num_mfcc=12)
    scores = [gmm.score(mfcc_test) for gmm in gmms]
    speaker_id = np.argmax(scores)
    return speaker_id

def main():
    # Define paths
    path = 'SpeakerData/'
    train_path = os.path.join(path, 'Train')
    test_path = os.path.join(path, 'Test')

    # Get the list of speakers
    speakers = sorted(os.listdir(train_path))
    print("Speakers:", speakers)

    # Create directories if they do not exist
    if not os.path.exists('TrainingFeatures'):
        os.makedirs('TrainingFeatures')
    if not os.path.exists('Models'):
        os.makedirs('Models')

    # Parameters
    hop_duration = 0.015  # 15ms
    num_mfcc = 12
    n_components = 5
    max_iter = 50

    # Step 1: Extract MFCC features and train GMMs for each speaker
    mfcc_all_speakers = []
    gmms = []

    for speaker in speakers:
        print(f"Processing speaker: {speaker}")
        # Path to the speaker's training audio files
        speaker_train_path = os.path.join(train_path, speaker)
        audio_files = [os.path.join(speaker_train_path, f) for f in os.listdir(speaker_train_path)]
        mfcc_one_speaker = np.asarray([])

        for audio_file in audio_files:
            # Extract MFCC features from each training audio file
            mfcc_one_file = mfcc_extraction(audio_file, hop_duration, num_mfcc)
            if mfcc_one_speaker.size == 0:
                mfcc_one_speaker = mfcc_one_file
            else:
                mfcc_one_speaker = np.vstack((mfcc_one_speaker, mfcc_one_file))

        # Save MFCC features to file
        with open(f'TrainingFeatures/{speaker}_mfcc.fea', 'wb') as f:
            pickle.dump(mfcc_one_speaker, f)

        mfcc_all_speakers.append(mfcc_one_speaker)

        # Train GMM model for the speaker
        gmm = learningGMM(mfcc_one_speaker, n_components, max_iter)
        gmms.append(gmm)

        # Save the GMM model to file
        with open(f'Models/{speaker}.gmm', 'wb') as f:
            pickle.dump(gmm, f)

    print("GMM training completed and models saved.")

    # Step 2: Perform speaker recognition on the test dataset
    print("Starting speaker recognition on the test dataset...")
    true_labels = []
    predicted_labels = []

    for speaker_id, speaker in enumerate(speakers):
        print(f"Testing speaker: {speaker}")
        speaker_test_path = os.path.join(test_path, speaker)
        test_files = [os.path.join(speaker_test_path, f) for f in os.listdir(speaker_test_path)]

        for test_file in test_files:
            # Recognize the speaker
            predicted_speaker_id = speaker_recognition(test_file, gmms)
            true_labels.append(speaker_id)
            predicted_labels.append(predicted_speaker_id)

            # Debug information
            print(f"Test file: {test_file}")
            print(f"True speaker: {speaker}")
            print(f"Predicted speaker: {speakers[predicted_speaker_id]}")
            print("----")

    # Calculate overall recognition accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall Recognition Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Display the confusion matrix
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(speakers))
    plt.xticks(tick_marks, speakers, rotation=45)
    plt.yticks(tick_marks, speakers)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    main()
