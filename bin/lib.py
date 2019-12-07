import subprocess
import pickle
import librosa
import cv2
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import datetime
import logging
import os
import pandas as pd
import yaml
from keras.preprocessing.sequence import pad_sequences
from gensim.utils import simple_preprocess
from collections import defaultdict

# Global variables
CONFS = None
BATCH_NAME = None

def extract_images(partition, num_frames):
    """
    Convert mp4 files into a series of jpeg images
    :param partition: Which data set
    :param num_frames: How many frames per video to extract
    """

    print('Begin image extraction on {} partition'.format(partition))

    # Define how many directories of videos to work through
    file_chunks = os.listdir('../data/video_data')
    file_chunks = [i for i in file_chunks if partition in i]

    for chunk in file_chunks:

        files = os.listdir('../data/video_data/{}'.format(chunk))

        for file_name in files:

            # Create video object
            cap = cv2.VideoCapture('../data/video_data/{}/{}'.format(chunk, file_name))

            # Get file name
            file_name = (file_name.split('.mp4'))[0]

            # Create new folder for images
            try:
                if not os.path.exists('../data/image_data/{}_data/{}'.format(partition, file_name)):
                    os.makedirs('../data/image_data/{}_data/{}'.format(partition, file_name))

            except OSError:
                logging.warning('Error: Creating directory of data')

            # Set number of frames to grab
            cap.set(cv2.CAP_PROP_FRAME_COUNT, num_frames + 1)
            length = num_frames + 1
            count = 0

            while cap.isOpened():
                count += 1

                # Exit if at the end
                if length == count:
                    break

                # Create the image
                ret, frame = cap.read()

                # Skip if there is no frame
                if frame is None:
                    continue

                # Resize image
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)

                # Save image to jpg
                name = '../data/image_data/{}_data/{}/frame{}.jpg'.format(partition, file_name, count)
                cv2.imwrite(name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                print('{}: {}: frame{}'.format(chunk, file_name, count))
    pass


def extract_audio(partition):
    """
    Extract mp3 files from each mp4 file
    :param partition: Which data set to use
    """

    print('Begin Audio Extraction on {} partition'.format(partition))

    # Define how many directories of videos to work through
    file_chunks = os.listdir('../data/video_data')
    file_chunks = [i for i in file_chunks if partition in i]

    # Create new folder for images
    if not os.path.exists('../data/audio_data/{}_data/'.format(partition)):
        os.makedirs('../data/audio_data/{}_data/'.format(partition))

    for chunk in file_chunks:

        files = os.listdir('../data/video_data/{}'.format(chunk))

        for file_name in files:
            file_name = file_name.split('.mp4')[0]

            # Use ffmpeg to strip audio from the video file
            subprocess.call(['ffmpeg',
                             '-y',
                             '-i',
                             '../data/video_data/{}/{}.mp4'.format(chunk, file_name),
                             '../data/audio_data/{}_data/{}.wav'.format(partition, file_name)])
    pass


def extract_text(partition):
    """
    Takes transcripts and saves them as dataframes
    :param partition: Which data set to use
    :param training: Is this run being used to train models or evaluate new videos
    :return:
    """

    training = False
    if partition == 'training':
        training = True

    print('Begin text extraction for {} partition'.format(partition))

    # Open transcript and annotations
    with open('../data/meta_data/transcription_{}.pkl'.format(partition), 'rb') as f:
        transcript = pickle.load(f, encoding='latin1')

    
    with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
        annotation = pickle.load(f, encoding='latin1')

    # Transform into a data frame
    text_df = pd.DataFrame({'video_id': list(transcript.keys()),
                            'transcript': list(transcript.values())})

    # When a video doesnt have a transcription, fill value with the unknown token
    text_df['transcript'] = text_df['transcript'].fillna('UNK')

    # Remove punctuation
    text_df['token'] = text_df['transcript'].str.replace(r'\[.*\]', '')

    
    text_df['interview_score'] = text_df['video_id'].map(annotation['interview'])
    print('Training is true!!!')

    # Create directory if it doesnt exist
    if not os.path.exists('../data/text_data/{}_data/'.format(partition)):
        os.makedirs('../data/text_data/{}_data/'.format(partition))

    # Save to disk
    with open('../data/text_data/{}_data/{}_text_df.pkl'.format(partition, partition), 'wb') as output:
        pickle.dump(text_df, output, protocol=4)

    pass


def transform_images(partition, num_frames):
    """
    Convert jpegs to numpy arrays and preprocess for the vgg16 model
    :param partition: Which data set to use
    :param num_frames: How many frames are contained in each numpy array
    :param training: Is this run for training models or evaluating new videos
    """
    training = False
    if partition == 'training':
        training = True

    print('Begin transform images 5d for the {} partition'.format(partition))

    if not os.path.exists('../data/image_data/npy_files/{}_data/'.format(partition)):
        os.makedirs('../data/image_data/npy_files/{}_data/'.format(partition))

    
    with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
        label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    vid_ids = os.listdir('../data/image_data/{}_data'.format(partition))
    file_ids = [i + '.mp4' for i in vid_ids]

    
    y = [label_file['interview'][i + '.mp4'] for i in vid_ids]

    out_counter = 0

    for video in vid_ids:

        images = os.listdir('../data/image_data/{}_data/{}'.format(partition, video))
        X_temp = np.zeros(shape=(num_frames, 224, 224, 3))
        in_counter = 0

        for image in images:

            # Load the image
            original = load_img('../data/image_data/{}_data/{}/{}'.format(partition, video, image),
                                target_size=(224, 224))

            # Convert to numpy array
            numpy_image = img_to_array(original)

            # Resize and store in one big array
            image_temp = np.expand_dims(numpy_image, axis=0)
            image_temp = vgg16.preprocess_input(image_temp)
            X_temp[in_counter] = image_temp

            # Increment counter for number of images in observation
            in_counter += 1

        # Append to numpy array
        X_temp = np.expand_dims(X_temp, axis=0)

        # Save the images numpy array
        np.save('../data/image_data/npy_files/{}_data/{}.npy'.format(partition, out_counter), X_temp)

        # Increment counter for observations in dataset
        out_counter += 1

    
    with open('../data/image_data/pickle_files/y_5d_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(y, output, protocol=4)


    # Save to disk
    with open('../data/image_data/pickle_files/vid_ids_5d_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(file_ids, output, protocol=4)

    pass


def transform_audio(partition, n_mfcc):
    """
    Use librosa to extract features and save dataframe with all features for each video
    :param partition: Which data set to use
    :param n_mfcc: Number of mfc coefficients to extract
    :param training: Is this run for training models or evaluating new videos
    """

    print('Begin audio transformations for {} partition'.format(partition))

    
    with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
        label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    audio_files = os.listdir('../data/audio_data/{}_data'.format(partition))
    audio_files = [i.split('.wav')[0] for i in audio_files]
    id_array = [i + '.mp4' for i in audio_files]

    
    score_interview = [label_file['interview'][i + '.mp4'] for i in audio_files]
    score_extraversion = [label_file['extraversion'][i + '.mp4'] for i in audio_files]
    score_agreeableness = [label_file['agreeableness'][i + '.mp4'] for i in audio_files]
    score_conscientiousness = [label_file['conscientiousness'][i + '.mp4'] for i in audio_files]
    score_neuroticism = [label_file['neuroticism'][i + '.mp4'] for i in audio_files]
    score_openness = [label_file['openness'][i + '.mp4'] for i in audio_files]
    
    score_interview = np.array(score_interview)
    score_extraversion = np.array(score_extraversion)
    score_agreeableness = np.array(score_agreeableness)
    score_conscientiousness = np.array(score_conscientiousness)
    score_neuroticism = np.array(score_neuroticism)
    score_openness = np.array(score_openness)

    # Set column names
    mfcc_mean_cols = ['mfcc_mean_' + str(i) for i in range(n_mfcc)]
    mfcc_std_cols = ['mfcc_mean_' + str(i) for i in range(n_mfcc)]
    other_cols = [

        'energey_mean',
        'energy_std',
        'zero_cross_mean',
        'zero_cross_std',
        'tempo_mean',
        'tempo_std',
        'flatness_mean',
        'flatness_std',
        'bandwidth_mean',
        'bandwidth_std',
        'rolloff_mean',
        'rolloff_std',
        'contrast_mean',
        'contrast_std',
        'tonnetz_mean',
        'tonnetz_std'
    ]
    cols = mfcc_mean_cols + mfcc_std_cols + other_cols

    # Create empty 2d array with place holders for all features
    audio_matrix = np.empty((len(audio_files), n_mfcc * 2 + 16))
    counter = 0

    for aud in audio_files:

        logging.debug('Begin feature extraction for {}.wav'.format(aud))

        # Convert wav to librosa object
        y, sr = librosa.load('../data/audio_data/{}_data/{}.wav'.format(partition, aud))

        # Create array to store values (Will become a row in the final df)
        values = np.zeros((len(cols)))

        mfcc = librosa.feature.mfcc(y, n_mfcc=n_mfcc)
        energy = librosa.feature.rms(y)
        zero_cross = librosa.feature.zero_crossing_rate(y)
        tempo = librosa.feature.tempogram(y)
        flatness = librosa.feature.spectral_flatness(y)
        bandwidth = librosa.feature.spectral_bandwidth(y)
        rolloff = librosa.feature.spectral_rolloff(y)
        contrast = librosa.feature.spectral_contrast(y)
        tonnetz = librosa.feature.tonnetz(y)

        values[0:n_mfcc] = mfcc.mean(axis=1)
        values[n_mfcc:n_mfcc * 2] = mfcc.std(axis=1)
        values[n_mfcc * 2] = np.mean(energy)
        values[n_mfcc * 2 + 1] = np.std(energy)
        values[n_mfcc * 2 + 2] = np.mean(zero_cross)
        values[n_mfcc * 2 + 3] = np.std(zero_cross)
        values[n_mfcc * 2 + 4] = np.mean(tempo)
        values[n_mfcc * 2 + 5] = np.std(tempo)
        values[n_mfcc * 2 + 6] = np.mean(flatness)
        values[n_mfcc * 2 + 7] = np.std(flatness)
        values[n_mfcc * 2 + 8] = np.mean(bandwidth)
        values[n_mfcc * 2 + 9] = np.std(bandwidth)
        values[n_mfcc * 2 + 10] = np.mean(rolloff)
        values[n_mfcc * 2 + 11] = np.std(rolloff)
        values[n_mfcc * 2 + 12] = np.mean(contrast)
        values[n_mfcc * 2 + 13] = np.std(contrast)
        values[n_mfcc * 2 + 14] = np.mean(tonnetz)
        values[n_mfcc * 2 + 15] = np.std(tonnetz)

        # Append values to matrix
        audio_matrix[counter] = values
        counter += 1

    # Create final dataframe
    audio_df = pd.DataFrame(audio_matrix, columns=cols)
    audio_df['interview_score'] = score_interview
    audio_df['extraversion'] = score_extraversion
    audio_df['agreeableness'] = score_agreeableness
    audio_df['conscientiousness'] = score_conscientiousness
    audio_df['neuroticism'] = score_neuroticism
    audio_df['openness'] = score_openness
    
    audio_df['video_id'] = id_array

    #print(id_array[0:20])

    audio_df.to_csv('../data/audio_data/pickle_files/{}_df_all.csv'.format(partition, partition), index=False)

    pass


def transform_text(partition, word_to_index):
    """
    Tokenize, and convert to indices based on the google news 20 word embeddings
    :param partition: Which data set to use
    :param word_to_index: dictionary that translates indices to vectors in the google news 20 word embeddings
    :param training: Is this run for training models or evaluating new videos
    """

    print('Begin text transformation on {}'.format(partition))

    # Load transcripts
    with open('../data/text_data/{}_data/{}_text_df.pkl'.format(partition, partition), 'rb') as f:
        observations = pickle.load(f, encoding='latin1')

    # Transform embedding resources
    default_dict_instance = defaultdict(lambda: word_to_index['UNK'])
    default_dict_instance.update(word_to_index)
    word_to_index = default_dict_instance

    # Convert text to normalized tokens. Unknown tokens will map to 'UNK'.
    observations['tokens'] = observations['transcript'].apply(simple_preprocess)

    # Convert tokens to indices
    observations['indices'] = observations['tokens'].apply(lambda token_list:
                                                           list(map(lambda token: word_to_index[token], token_list)))
    observations['indices'] = observations['indices'].apply(lambda x: np.array(x))

    # Pad indices list with zeros, so that every article's list of indices is the same length
    X = pad_sequences(observations['indices'], 80)

    cols = observations.columns.tolist()
    print(cols)

    # Create data sets for model
    
    y_interview = observations['interview_score'].values
    y_extraversion = observations['extraversion'].values
    y_agreeableness = observations['agreeableness'].values
    y_conscientiousness = observations['conscientiousness'].values
    y_neuroticism = observations['neuroticism'].values
    y_openness = observations['openness'].values

    y = np.zeros((len(y_interview), 6))
    for i in range(len(y_interview)):
        y[i][0] = y_interview[i]
        y[i][1] = y_extraversion[i]
        y[i][2] = y_agreeableness[i]
        y[i][3] = y_conscientiousness[i]
        y[i][4] = y_neuroticism[i]
        y[i][5] = y_openness[i]

    vid_id = observations['video_id'].values

    # Save as pickled files
    with open('../data/text_data/pickle_files/X_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(X, output, protocol=4)
    with open('../data/text_data/pickle_files/vid_ids_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(vid_id, output, protocol=4)

    with open('../data/text_data/pickle_files/y_{}_all.pkl'.format(partition), 'wb') as output:
        pickle.dump(y, output, protocol=4)

    pass
