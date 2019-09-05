# Basic CNN (using Keras) following tutorial at
#     https://github.com/ajhalthor/audio-classifier-convNet
# Machine learning library: keras

import numpy as np # following all used by train_simple_keras
import keras 
from   keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from   keras.models import Sequential
from   keras.models import model_from_json
from   keras import backend as K 
import librosa
import librosa.display
import pandas as pd 
import random 
import warnings
warnings.filterwarnings('ignore')

import os
import sys 
sys.path.insert(0, '/Users/hzd88688126com/Desktop/Project/Code')
import wavtools # contains custom functions e.g. denoising

from   pydub import AudioSegment # used by: search_file_
from   pydub.utils import make_chunks # used by: search_file_
from   shutil import copyfile # used by: search_file_
from   shutil import rmtree # used by: search_file_
import glob # used by: search_file_, search_folder_, hard_negative_miner
import subprocess # used by: search_file_
import time # used by: search_folder_
import pathlib # used by: search_file_
import csv # used by: search_file_
import matplotlib.pyplot as plt # used by train_simple_
from   sklearn.preprocessing import LabelEncoder # added for single output layer
from   sklearn.preprocessing import scale
from   sklearn.metrics import roc_curve # used in train_simple_
from   sklearn.metrics import auc

import functools    # these three used in as_keras_metric
from   keras import backend as K
import tensorflow as tf

# ## NB.! time per file is approx. 1.553 seconds. 
# ## resulting in time per folder (~4000 files) of approx. 1.75 hours 


# ########################## REMOVE THIS SECTION WHEN I'VE STOPPED EXPERIMENTING WITH IT ####################################################
####### - added so functions and application of functions in same script, preventing having to import updated functions every time ##########

import sys
import pickle

sys.path.insert(0, '/Users/hzd88688126com/Desktop/Project/Data/Code')

praat_files= sorted([f for f in os.listdir('../Data/praat-files') if not f.startswith('.')])
# RUN 1 - no preprocessing
# =============================================================================
# # compile dataset
# D_original = wavtools.compile_dataset('without-preprocessing',sr=44100)
# 
# # RUN 2 - denoised only, no augmentations
# # compile dataset
# D_denoised = wavtools.compile_dataset('denoised',sr=44100)
# 
# # RUN 3 - denoised, Gaussian noise augmentation added
# # compile dataset
# D_denoised_noise_aug = wavtools.compile_dataset('denoised/noise-aug',sr=44100)
# 
# # RUN 4 - denoised, unbalanced classes (all known negatives)
# # compile dataset
# D_denoised_noise_aug = wavtools.compile_dataset('denoised/noise-aug',sr=44100)
# 
# # RUN 5 - denoised, random crop augmentation, unbalanced classes (all known negatives)
# # compile dataset
# D_denoised_crop_aug_unbalanced = wavtools.compile_dataset('denoised/crop-aug/unbalanced',sr=44100)
# =============================================================================

# training for a given run_type:
#train_perc = 0.8
#batch_size = 16
#num_epochs = 10
#sr = 44100
## dataset = D_original
#name = 'D_original'
# train_simple_keras_ONE_NODE_OUTPUT(dataset,name,train_perc,num_epochs,batch_size,sr)
# train_simple_keras(dataset,name,train_perc,num_epochs,batch_size)

# BELOW WAS WORKFLOW BEFORE I CAME UP WITH COMPILE DATA FUNCTION,
# WHICH IS WHAT I USED ABOVE

# # # method 5: adding hard-negative mined training examples 

# # # # hard_negative_miner('/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/', 62, model=loaded_model)
# # # D_mined_aug_tb = D_aug_tb 
# # # wavtools.add_files_to_dataset(folder='mined-false-positives', dataset=D_mined_aug_tb, example_type=0)

# # # print("\nNumber of samples when hard negatives added: " + str(wavtools.num_examples(D_mined_aug_tb,0)) + \
# # # " negative, " + str(wavtools.num_examples(D_mined_aug_tb,1)) + " positive"))

# # # D_mined_aug_tb_denoised = wavtools.denoise_dataset(D_mined_aug_tb)

# # # method 6: adding selected obvious false positives as training examples

# # D_S_mined_aug_t_denoised = D_aug_t

# # wavtools.add_files_to_dataset(folder='selected-false-positives', dataset=D_S_mined_aug_t_denoised, example_type=0)

# # print("\nNumber of samples when select negatives added: " + str(wavtools.num_examples(D_S_mined_aug_t_denoised,0)) + \
# # " negative, " + str(wavtools.num_examples(D_S_mined_aug_t_denoised,1)) + " positive"))

# # # method 7: adding 'most wrong' false positives as training examples

# # # tried ~100 negatives from Catappa, ~100 positives that I had 
# # # DID NOT WORK. great results but background noise between positives and negatives was too different to generalise
# # # workflow was:
# # D_MW_mined = []
# # wavtools.add_files_to_dataset(folder='clipped-whinnies', dataset=D_MW_mined, example_type=1)
# # wavtools.add_files_to_dataset(folder='selected-false-positives/from-unclipped-whinnies', dataset=D_MW_mined, example_type=0)
# # wavtools.add_files_to_dataset(folder='selected-false-positives/catappa2-from-jenna', dataset=D_MW_mined, example_type=0)
# # D_MW_mined_denoised = wavtools.denoise_dataset(D_MW_mined)

# ### VIEWING SPECTROGRAMS 
# wav = '../Data/clipped-whinnies/5A3844FE_1.WAV'
# mag_spec = wavtools.load_mag_spec(wav, sr=41000, denoise=False, normalize=False)
# # different mel.spec generating methods
# mel_spec = D_denoised[24][0]
# mel_spec = wavtools.do_augmentation(mag_spec, sr=41000, noise=False, noise_samples=False, roll=False)

def view_mag_spec(mag_spec):
    librosa.display.specshow(librosa.power_to_db(mag_spec),x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Magnitude spectrogram')
    plt.show()

def view_mel_spec(mel_spec, save=False):
    librosa.display.specshow(librosa.power_to_db(mel_spec),y_axis='mel',x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    if save:
        plt.title('Mel spectrogram')
        wavname = os.path.basename(os.path.splitext(wav)[0])
        plt.savefig('../Results/viewing-input-spects/'+wavname+'.png')
        plt.gcf().clear()
    else:
        plt.show()

def view_mag_and_mel(mel_spec, folder):
    # mel
    mel_spec = np.flip(mel_spec,axis=0)
    plt.imshow(mel_spec)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    wavname = os.path.basename(os.path.splitext(wav)[0])
    plt.savefig('../Results/viewing-input-spects/'+wavname+'.png')
    plt.gcf().clear()

# view_mag_and_mel(mag_spec, mel_spec)

# can I cut the input size...
# viewing all spects for the positives we have:
# folder = 'clipped-whinnies'
# data_folder_path = '../Data/'
# files = glob.glob(data_folder_path+folder+'/*.WAV')
# for wav in files:
#     mag_spec = wavtools.load_mag_spec(wav, sr=41000, denoise=False, normalize=False)
#     mel_spec = wavtools.do_augmentation(mag_spec, sr=41000, noise=False, noise_samples=False, roll=False)
#     view_mag_and_mel(mag_spec,mel_spec)

# AVERAGE CALL LENGTH OF ALL CALLS RECORDED

call_durations = []

for wav in praat_files:
    try:
	    start_times = wavtools.whinny_starttimes_from_praatfile('../Data/praat-files/'+wav)[1]
    except FileNotFoundError:
        print("Unable to process file:", wav)
        continue
    end_times = wavtools.whinny_endtimes_from_praatfile('../Data/praat-files/'+wav)[1]

    call_durations.extend([ends-starts for starts,ends in zip(start_times, end_times)])

avg_call_len = sum(call_durations) / float(len(call_durations))
longest_call = np.max(call_durations)
shortest_call = np.min(call_durations)

#################################################################
## MISC. CODE. NOT USED AT PRESENT

def custom_recall(y_true, y_pred, threshold):
    """Recall metric, with threshold option
    """
    y_true = np.ndarray.tolist(y_true)
    y_pred = np.ndarray.tolist(y_pred)
    # flatten list of lists to list
    y_pred = [item for sublist in y_pred for item in sublist]
    # round predictions to 0 or 1
    for idx, val in enumerate(y_pred):
        y_pred[idx] = 0 if val < threshold else 1 
    # true positives equals num. that sum to 2 ('1' and '1')
    summed = [x+y for x, y in zip(y_true, y_pred)]
    true_positives = sum(i == 2 for i in summed)
    # loop to find false positives
    possible_positives = sum(i == 1 for i in y_true)
    recall = true_positives/(true_positives+possible_positives)
    return recall 

def custom_precision(y_true, y_pred, threshold):
    """Precision metric, with threshold option
    """
    y_true = np.ndarray.tolist(y_true)
    y_pred = np.ndarray.tolist(y_pred)
    # flatten list of lists to list
    y_pred = [item for sublist in y_pred for item in sublist]
    # round predictions to 0 or 1
    for idx, val in enumerate(y_pred):
        y_pred[idx] = 0 if val < threshold else 1 
    # true positives equals num. that sum to 2 ('1' and '1')
    summed = [x+y for x, y in zip(y_true, y_pred)]
    true_positives = sum(i == 2 for i in summed)
    # loop to find false positives
    false_positives = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 0:
            false_positives += 1 
    precision = true_positives/(true_positives+false_positives)
    return precision 

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        Taken from https://stackoverflow.com/questions/42606207
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        Taken from: https://stackoverflow.com/questions/42606207
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

# following section taken from https://stackoverflow.com/questions/6392739
# in attempt to troubleshoot metric value problems (prev. too high)
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
@as_keras_metric
def auc_pr(y_true, y_pred, curve='PR'):
    return tf.metrics.auc(y_true, y_pred, curve=curve)



def search_folder_for_monkeys(wav_folder, threshold_confidence, model, sr):
    
    # list all file names in folder
    wavs = glob.glob(wav_folder+'*.WAV')
    wavs = [os.path.splitext(x)[0] for x in wavs]
    wavs = [os.path.basename(x) for x in wavs]

    # require user input if code is suspected to take long to run
    predicted_run_time = len(wavs)*1.553
    if len(wavs) > 30:
        confirmation = input("\nWarning: this code will take approximately " + str(round(predicted_run_time/60, 3)) + " minutes to run. enter Y to proceed\n\n")
        if confirmation != "Y":
            print('\nError: function terminating as permission not received')
            return 
    tic = time.time()

    for wav in wavs:
        search_file_for_monkeys_ONE_NODE_OUTPUT(wav, threshold_confidence, wav_folder, model, sr, full_verbose=False, summary_file=True)

    toc = time.time()
    print('\nSystem took', round((toc-tic)/60, 3), 'mins to process', len(wavs), 'files\n\nfor a summary of results, see the csv file created in Results folder\n')

def hard_negative_miner(wav_folder, threshold_confidence, model):

    # list all file names in folder
    wavs = glob.glob(wav_folder+'*.WAV')
    wavs = [os.path.splitext(x)[0] for x in wavs]
    wavs = [os.path.basename(x) for x in wavs]

    for wav in wavs:
        search_file_for_monkeys_ONE_NODE_OUTPUT(wav, threshold_confidence=threshold_confidence, wav_folder=wav_folder, sr=sr, model=model, hnm=True)

def search_file_for_monkeys_ONE_NODE_OUTPUT(file_name, threshold_confidence, wav_folder, model, sr, denoise=False, standardise=False, tidy=True, full_verbose=True, hnm=False, summary_file=False):
    """
    Splits 60-second file into 3-second clips. Runs each through
    detector. If activation surpasses confidence threshold, clip
    is separated.
    If hard-negative mining functionality selected, function
    takes combination of labelled praat file and 60-second wave file,
    runs detector on 3-second clips, and seperates any clips that 
    the detector incorrectly identifies as being positives.
    These clips are then able to be fed in as negative examples, to
    improve the discriminatory capability of the network 

    Example call: 
    search_file_for_monkeys_ONE_NODE_OUTPUT('5A3BE710', 60, '/Users/hzd88688126com/Desktop/Project/Data/unclipped-whinnies/', loaded_model)

    NB. use following for testing, but REMEMBER TO DELETE***********
    file_name = '5A3BE710'
    threshold_confidence = 70
    wav_folder = '/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/'
    model = loaded_model

    """

    # increased sampling rate gives increased number of time-slices per clip
    # this affects CNN input size, and time_dimension used as proxy to ensure 
    # all clips tested are of the same length (if not, they are not added to test dataframe)
    if sr == 44100:
        time_dimension = 299
    elif sr == 48000:
        time_dimension = 299
    else:
        return("Error: sampling rate must be 48000 or 44100") 

    # isolate folder name from path:
    p = pathlib.Path(wav_folder)
    isolated_folder_name = p.parts[2:][-1]
    wav = wav_folder+file_name+'.WAV'
    # checks: does audio file exist and can it be read
    if not os.path.isfile(wav):
        print("\nError: no audio file named",os.path.basename(wav),"at path", os.path.dirname(wav))
        return 
    try:
        wavfile = AudioSegment.from_wav(wav)
    except OSError:
        print("\nError: audio file",os.path.basename(wav),"at path",os.path.dirname(wav),"exists but cannot be loaded (probably improperly recorded)")
        return 
    clip_length_ms = 3000
    clips = make_chunks(wavfile, clip_length_ms)

    print("\n-- processing file " + file_name +'.WAV')

    # if hard-negative mining, test for presence of praat file early for efficiency:
    if hnm:
        praat_file_path = '../Data/praat-files/'+file_name+'.TextGrid'
        try:
            labelled_starts = wavtools.whinny_starttimes_from_praatfile(praat_file_path)[1]

        except IOError:
            print('Error: no praat file named',os.path.basename(praat_file_path),'at path', os.path.dirname(praat_file_path))
            return

    clip_dir = wav_folder+'clips-temp/'

    # delete temporary clips directory if interuption to previous
    # function call failed to remove it 
    if os.path.exists(clip_dir) and os.path.isdir(clip_dir):
        rmtree(clip_dir)
    # create temporary clips directory 
    os.makedirs(clip_dir) 
    # export all inviduals clips as wav files
    # print('clipping 60 second audio file into 3 second snippets to test...\n')
    for clipping_idx, clip in enumerate(clips):
        clip_name = "clip{0:02}.wav".format(clipping_idx+1)
        clip.export(clip_dir+clip_name, format="wav")

    clipped_wavs = glob.glob(clip_dir+'clip*')
    clipped_wavs = sorted(clipped_wavs, key=lambda item: (int(item.partition(' ')[0])
                                if item[0].isdigit() else float('inf'), item))

    # preallocate dataframe of correct length
    D_test = np.zeros(len(clipped_wavs), dtype=object)

    for data_idx, clip in enumerate(clipped_wavs):
        # y, sr = librosa.load(clip, sr=sr, duration=3.00)
        mag_spec = wavtools.load_mag_spec(clip, sr, denoise=denoise, normalize=False)
        ps = librosa.feature.melspectrogram(S=mag_spec, sr=sr)
        if ps.shape != (128, time_dimension): continue
        D_test[data_idx] = ps

    # # conditions for modifying file:
    # if denoise == True:
    #     D_test = wavtools.denoise_dataset(D_test)
    # if standardise == True:
    #     D_test = wavtools.standardise_inputs(D_test)

    # counters for informative naming of files
    call_count = 0
    hnm_counter = 0

    # reshape to be correct dimension for CNN input
    # NB. dimensions are: num.samples, num.melbins, num.timeslices, num.featmaps 
    # print("...checking clips for monkeys...")
    for idx, clip in enumerate(D_test):
        D_test[idx] = clip.reshape(1,128,time_dimension,1)
        predicted = model.predict(D_test[idx])

        # if NEGATIVE:
        if predicted[0][0] <= (threshold_confidence/100.0): ########## THIS IS SECTION THAT CHANGED BETWEEN 1 node/2 node:
            continue                                        # WAS: if predicted[0][1] <= (threshold_confidence/100.0)
                                                            # furthermore 3 changes (predicted[0][1] -> ..cted[0][0]) below            
        else:
        # if POSITIVE
            call_count+=1
            lower_clip_bound = (3*(idx+1))-3
            upper_clip_bound = 3*(idx+1)
            # i.e. clip 3 would be 6-9 seconds into original 60-sec file
            approx_position = str(lower_clip_bound)+'-'+str(upper_clip_bound)

            # regular detector behaviour - not hard negative mining
            if not hnm:
                # suspected positives moved to folder in Results, files renamed 'filename_numcallinfile_confidence.WAV'
                # results_dir = '/media/dgabutler/My Passport/Audio/detected-positives/'+isolated_folder_name+'-results'
                results_dir = '/Users/hzd88688126com/Desktop/Project/Results/detected-positives/'+isolated_folder_name+'-results'

                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                copyfile(clipped_wavs[idx], results_dir+'/'+file_name+'_'+str(call_count)+'_'+approx_position+'_'+str(int(round(predicted[0][0]*100)))+'.WAV')

                # making summary file 
                if summary_file:
                    summary_file_name = '/Users/hzd88688126com/Desktop/Project/Results/'+isolated_folder_name+'-results-summary.csv'
                    # obtain datetime from file name if possible 
                    try:
                        datetime_of_recording = wavtools.filename_to_localdatetime(file_name)
                        date_of_recording = datetime_of_recording.strftime("%d/%m/%Y")
                        time_of_recording = datetime_of_recording.strftime("%X")
                    # if not possible due to unusual file name, 
                    # assign 'na' value to date time 
                    except ValueError:
                        date_of_recording = 'NA'
                        time_of_recording = 'NA' 
                    
                    # values to be entered in row of summary file:
                    column_headings = ['file name', 'approx. position in recording (secs)', 'time of recording', 'date of recording', 'confidence']
                    csv_row = [file_name, approx_position, time_of_recording, date_of_recording, str(int(round(predicted[0][0]*100)))+'%']
                        
                    # make summary file if it doesn't already exist
                    summary_file_path = pathlib.Path(summary_file_name)
                    if not summary_file_path.is_file():
                        with open(summary_file_name, 'w') as csvfile:
                            filewriter = csv.writer(csvfile, delimiter=',')
                            filewriter.writerow(column_headings)
                            filewriter.writerow(csv_row)
                    
                    # if summary file exists, *append* row to it
                    else:
                        with open(summary_file_name, 'a') as csvfile:
                            filewriter = csv.writer(csvfile, delimiter=',')
                            filewriter.writerow(csv_row)
            else:
            # if hard-negative mining for false positives to enhance training:
                labelled_ends = wavtools.whinny_endtimes_from_praatfile(praat_file_path)[1]

                if not any(lower_clip_bound <= starts/1000.0 <= upper_clip_bound for starts in labelled_starts) \
                and not any(lower_clip_bound <= ends/1000.0 <= upper_clip_bound for ends in labelled_ends):                   
                    # i.e. if section has not been labelled as containing a call
                    # (therefore a false positive has been detected)
                    hnm_counter+=1
                    copyfile(clipped_wavs[idx], '/Users/hzd88688126com/Desktop/Project/Data/mined-false-positives/'+file_name+'_'+str(hnm_counter)+'_'+approx_position+'_'+str(int(round(predicted[0][0]*100)))+'.WAV')
                else: continue     

        # if full_verbose:
        #     print('clip number', '{0:02}'.format(idx+1), '- best guess -', best_guess)

    # delete all created clips and temporary clip folder
    if tidy:
        rmtree(clip_dir)
        # empty recycling bin to prevent build-up of trashed clips
        subprocess.call(['rm -rf /Users/hzd88688126com/.local/share/Trash/*'], shell=True)

    # print statements to terminal
    if full_verbose:
        if not hnm:
            print('\nfound', call_count, 'suspected call(s) that surpass %d%% confidence threshold in 60-second file %s.WAV' % (threshold_confidence, file_name))
        else:
            print('\nhard negative mining generated', hnm_counter, 'suspected false positive(s) from file', file_name, 'for further training of network')
def train_simple_keras(dataset, name, train_perc, num_epochs, batch_size):
    """
    Trains and saves simple keras model. 
    """
    try:
        random.shuffle(dataset)
    except NameError:
        print('non-existent dataset name provided. check dataset exists and retry')
        return 

    # use provided training percentage to give num. training samples
    n_train_samples = int(round(len(dataset)*train_perc))
    train = dataset[:n_train_samples]
    # tests on remaining % of total
    test = dataset[n_train_samples:]    

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # reshape for CNN input
    X_train = np.array([x.reshape( (128, 299, 1) ) for x in X_train])
    X_test = np.array([x.reshape( (128, 299, 1) ) for x in X_test])

    # one-hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, 2))
    y_test = np.array(keras.utils.to_categorical(y_test, 2))

    model = Sequential()
    input_shape=(128, 299, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.6152916582980337)) # from hyperas

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.23855852860918042)) # from hyperas

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # following two lines are experimenting with diff. metric method
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=['accuracy', precision, recall]) 

    history = model.fit( # was 'model.fit('
        x=X_train, 
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data= (X_test, y_test))

    score = model.evaluate(
        x=X_test,
        y=y_test)
    print('Test accuracy:', score[1])
    
def train_simple_keras_ONE_NODE_OUTPUT(dataset, name, train_perc, num_epochs, batch_size, sr):
    """
    Adapted train-and-save function, with
    SINGLE SIGMOID OUTPUT LAYER replacing the two nodes
    - decision taken following advice of Harry Berg, w/
    guidance from https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
    """
    try:
        random.shuffle(dataset)
    except NameError:
        print('Non-existent dataset name provided. Check dataset exists and retry')
        return 

    # increased sampling rate gives increased number of time-slices per clip
    # this affects CNN input size, and time_dimension used as proxy to ensure 
    # all clips tested are of the same length (if not, they are not added to test dataframe)
    if sr == 44100:
        time_dimension = 299
    elif sr == 48000:
        time_dimension = 299
    else:
        return("error: sampling rate must be 48000 or 44100") 

    # use provided training percentage to give num. training samples
    n_train_samples = int(round(len(dataset)*train_perc))
    train = dataset[:n_train_samples]
    # tests on remaining % of total
    test = dataset[n_train_samples:]    

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # reshape for CNN input
    X_train = np.array([x.reshape((128, time_dimension, 1)) for x in X_train])
    X_test = np.array([x.reshape((128, time_dimension, 1)) for x in X_test])

    # ALTERED ENCODING SECTION #######################################
    # previously was:
    # # one-hot encoding for classes
    # y_train = np.array(keras.utils.to_categorical(y_train, 2))
    # y_test = np.array(keras.utils.to_categorical(y_test, 2))

    # changed to...
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder.fit(y_test)
    encoded_y_train = encoder.transform(y_train)
    encoded_y_test = encoder.transform(y_test)
    ##################################################################

    model = Sequential()
    input_shape=(128, time_dimension, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5)) # can optimise

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5)) # can optimise

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # following two lines are experimenting with diff. metric method
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        # metrics=['accuracy', precision_threshold(0.5), recall_threshold(0.5)]) 
        metrics=['accuracy', precision, recall, auc_pr])

    history = model.fit( 
        x=X_train, 
        y=encoded_y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data= (X_test, encoded_y_test))

    score = model.evaluate(
        x=X_test,
        y=encoded_y_test)
    print('Test accuracy:', score[1])

    # # list all data in history
    # print(history.history.keys()) # added 
    print('\n')

    # custom function to create plots of training behaviour
    def training_behaviour_plot(metric):
        """
        Produces and saves plot of given metric for training
        and test datasets over duration of training time

        e.g. training_behaviour_plot('recall')
        """
        # check/make savepath 
        plot_save_path = '../Results/CNN-learning-behaviour-plots/one-node-output/'+'model_'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        # compile and save plot for given metric  
        file_name = plot_save_path+'/'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)+'_'+metric+'.png'
        # remove plot if already exists, to allow to be overwritten:
        if os.path.isfile(file_name):
            os.remove(file_name)
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title('model '+name+''+'_e'+str(num_epochs)+'_b'+str(batch_size)+' '+metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(file_name)
        plt.gcf().clear()
        print('saved '+metric+' plot to ../Results/CNN-learning-behaviour-plots/')

    training_behaviour_plot('acc')
    training_behaviour_plot('loss')
    training_behaviour_plot('recall')
    training_behaviour_plot('precision')

#################################################################
# MORE COMPLEX CONVNET WITH SPECIFIC IMPORT STATEMENTS
# from https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%202b%20-%20Train%20and%20Predict%20on%20UrbanSound%20dataset.ipynb

## NB! CURRENTLY TAKES MORE TIME AND IS LESS ACCURATE
## --- was abandoned early on so code may need edits to run 

# # change the seed before anything else
# import numpy as np
# np.random.seed(1)
# import tensorflow as tf
# tf.set_random_seed(1)

# import os
# import time

# import keras
# keras.backend.clear_session()

# import matplotlib.pyplot as plt
# import sklearn

# from keras.models import Sequential
# from keras.layers import Activation
# from keras.layers import Convolution2D, MaxPooling2D, Dropout
# from keras.layers.pooling import GlobalAveragePooling2D
# from keras.optimizers import Adamax
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.regularizers import l2
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# from keras.layers.normalization import BatchNormalization

# def train_complex_keras(dataset, train_perc):
#     """
#     Trains and more complex keras model. 

#     Taken from https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%202b%20-%20Train%20and%20Predict%20on%20UrbanSound%20dataset.ipynb

#     """
#     try:
#         random.shuffle(dataset)
#     except NameError:
#         print('non-existent dataset name provided. check dataset exists and retry')
#         return 

    # # use provided training percentage to give num. training samples
    # n_train_samples = int(round(len(dataset)*train_perc))
    # train = dataset[:n_train_samples]
    # # tests on remaining % of total
    # test = dataset[n_train_samples:]    

    # X_train, y_train = zip(*train)
    # X_test, y_test = zip(*test)

    # # reshape for CNN input
    # X_train = np.array([x.reshape( (128, 279, 1) ) for x in X_train])
    # X_test = np.array([x.reshape( (128, 279, 1) ) for x in X_test])

    # # one-hot encoding for classes
    # y_train = np.array(keras.utils.to_categorical(y_train, 2))
    # y_test = np.array(keras.utils.to_categorical(y_test, 2))

    # model = Sequential()
    # input_shape=(128, 279, 1)

    # # section 1

    # model.add(Convolution2D(filters=32, kernel_size=5,
    #                         strides=2,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal",
    #                         input_shape=input_shape))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=32, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.3))

    # # section 2    
    # model.add(Convolution2D(filters=64, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=64, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # # section 3
    # model.add(Convolution2D(filters=128, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=128, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=128, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=128, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # # section 4
    # model.add(Convolution2D(filters=512, kernel_size=3,
    #                         strides=1,
    #                         padding="valid",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    # model.add(Convolution2D(filters=512, kernel_size=1,
    #                         strides=1,
    #                         padding="valid",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    # # section 5
    # model.add(Convolution2D(filters=2, kernel_size=1,
    #                         strides=1,
    #                         padding="valid",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(GlobalAveragePooling2D())

    # model.add(Activation('softmax'))

    # model.compile(
    #     optimizer="Adam",
    #     loss="binary_crossentropy",
    #     metrics=['accuracy'])

    # model.fit(
    #     x=X_train, 
    #     y=y_train,
    #     epochs=12,
    #     batch_size=128,
    #     validation_data= (X_test, y_test))

    # score = model.evaluate(
    #     x=X_test,
    #     y=y_test)

    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # serialise model to JSON
    # dataset_name = name
    # save_path = '/home/dgabutler/Work/CMEEProject/Models/'+dataset_name+'/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # model_json = model.to_json()
    # with open(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.json', 'w') as json_file:
    #     json_file.write(model_json)
    # # serialise weights to HDF5
    # model.save_weights(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.h5')
    # print('\nsaved model '+dataset_name+'/'+'e'+str(num_epochs)+'_b'+str(batch_size)+' to disk')
