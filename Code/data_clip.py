import numpy as np 
import os 
import sys
import glob

sys.path.insert(0, '/Users/hzd88688126com/Desktop/Project/Code')
import wavtools   # contains custom functions e.g. denoising
import keras_classifier

###### DATA PROCESSING
# check target folders are empty, if script is re-run:
# clipped folders:
files = glob.glob('../Data/clipped-whinnies/*')
for f in files:
    os.remove(f)
files = glob.glob('../Data/clipped-negatives/*')
for f in files:
    os.remove(f)
# augmented folder:
files = glob.glob('../Data/aug-timeshifted/*')
for f in files:
    os.remove(f)
# clipping positives:'
praat_file= sorted([f for f in os.listdir('../Data/praat-files') if not f.startswith('.')])
############## Positives############
# LOCATION 1: OSA
# positives:
wavtools.clip_whinnies(praat_file, 3000, '../Data/Positives/osa','../Data/clipped-whinnies')
# LOCATION 2: SHADY
# positives:
wavtools.clip_whinnies(praat_file, 3000, '../Data/Positives/shady','../Data/clipped-whinnies')
# LOCATION 3: Other
# positives:
wavtools.clip_whinnies(praat_file, 3000, '../Data/Positives/other','../Data/clipped-whinnies')
# LOCATION 4: CATAPPA
# positives:
wavtools.clip_whinnies(praat_file, 3000, '../Data/Positives/catappa','../Data/clipped-whinnies')
# LOCATION 5: Live-recording
# positives:
wavtools.clip_whinnies(praat_file, 3000, '../Data/Positives/live-recording','../Data/clipped-whinnies')
# LOCATION 6: Will
# positives:
wavtools.clip_whinnies(praat_file, 3000, '../Data/Positives/will','../Data/clipped-whinnies')
# LOCATION 7: tape
# positives:
wavtools.clip_whinnies(praat_file, 3000, '../Data/Positives/tape','../Data/clipped-whinnies')

wavtools.clip_whinnies(praat_file, 3000, '../Data/Positives/new-calls','../Data/clipped-whinnies')

############## Negatives ##############
wavtools.clip_noncall_sections(praat_file,'../Data/Positives/osa')
wavtools.clip_noncall_sections(praat_file,'../Data/Positives/shady')
wavtools.clip_noncall_sections(praat_file,'../Data/Positives/other')
wavtools.clip_noncall_sections(praat_file,'../Data/Positives/catappa')
wavtools.clip_noncall_sections(praat_file,'../Data/Positives/live-recording')
wavtools.clip_noncall_sections(praat_file,'../Data/Positives/will')
wavtools.clip_noncall_sections(praat_file,'../Data/Positives/tape')
noncall_file= glob.glob('../Data/sections-without-whinnies/*.WAV')
wavtools.generate_negative_examples(noncall_file, 3)
noncall_file= glob.glob('../Data/Negatives-New/*.WAV')
wavtools.generate_negative_examples(noncall_file, 3)
###
