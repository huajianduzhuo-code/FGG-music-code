# for each file in no_drum_sample, convert to piano roll, perform segmentation, and save to create the dataset

import pretty_midi
import numpy as np

import glob
import tqdm
import os



PIANO = (0,1)


def midi_to_piano_roll(midi_file_path, resolution=16): # resolution=x means x-th note (1,2,4,8,16,32) 
    '''
    convert to midi file into piano_rolls of onset and sustain, each roll has shape L*128
    '''
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    end_time = midi_data.get_end_time()
    end_time_in_beats = midi_data.time_to_tick(end_time) / midi_data.resolution
    # print(end_time, end_time_in_beats, midi_data.resolution)

    piano_roll_len = int(np.ceil(end_time_in_beats * resolution/4))

    piano_rolls = []
    # Populate the piano roll
    # generate the piano rolls for all tracks
    for instrument in midi_data.instruments:
        if not (int(instrument.program) in PIANO): # the instrument is not in the set of piano
            #print('not piano')
            continue
        if instrument.is_drum:
            continue

        onset_roll = np.zeros((128, piano_roll_len), dtype=np.int32)
        sustain_roll = np.zeros((128, piano_roll_len), dtype=np.int32)
        notes_list = [] # each element is a list of three elements: [note_start, pitch, duration]

        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            start_beat = midi_data.time_to_tick(start_time) / midi_data.resolution
            end_beat = midi_data.time_to_tick(end_time) / midi_data.resolution
            start_step = int(np.round(start_beat * resolution/4))
            end_step = int(np.round(end_beat * resolution/4))
            if end_step == start_step and end_beat > start_beat: # deal with notes that are shorter than 16th note
                end_step = start_step + 1
            # print(start_time, start_beat, end_time, end_beat, start_step, end_step)
            if start_step>=end_step or start_step>=piano_roll_len:
                continue
            duration = end_step - start_step
            onset_roll[note.pitch, start_step] = 1
            sustain_roll[note.pitch, start_step+1:end_step] = 1
            notes_list.append([start_step, note.pitch, duration])

        piano_rolls.append({"onset_roll": np.array(onset_roll), "sustain_roll": np.array(sustain_roll), "notes_list": notes_list})

    return piano_rolls

def note_matrix_to_piano_roll(note_mat, total_length=None):
    total_length = total_length if total_length is not None else max(note_mat[:, 0] + note_mat[:, 2])

    piano_roll = np.zeros((2, total_length, 128), dtype=np.int64)
    for note in note_mat:
        onset, pitch, duration = note
        piano_roll[0, onset, pitch] = 1
        piano_roll[1, onset + 1: onset + duration, pitch] = 1
    return piano_roll


def combine_piano_roll(piano_rolls, resolution=16):
    # remove tracks with mean pitch < highest one - 12
    onset_rolls = []
    sustain_rolls = []
    
    for i in range(len(piano_rolls)):
        # if mean_pitch[i]>highest_mean_pitch-12:
            onset_rolls.append(piano_rolls[i]["onset_roll"])
            sustain_rolls.append(piano_rolls[i]["sustain_roll"])

    piano_roll = []
    onset_roll_combined = np.maximum.reduce(onset_rolls)
    sustain_roll_combined = np.maximum.reduce(sustain_rolls)
    #piano_roll = np.stack((onset_roll_combined.T, sustain_roll_combined.T))
    piano_roll.append({"onset_roll": onset_roll_combined, "sustain_roll": sustain_roll_combined})
    return piano_roll


def find_first_nonempty_column(arr):
    for col in range(arr.shape[1]):
        if not np.all(arr[:, col] == 0):
            return col
    return -1  # If empty piano_roll

def find_last_nonempty_column(arr):
    for col in range(arr.shape[1] - 1, -1, -1):
        if not np.all(arr[:, col] == 0):
            return col
    return -1  # If all columns are empty


# gather the nonempty measures

def find_first_nonempty_measure_beat(piano_roll, resolution=16): # resolution=x means x-th note (1,2,4,8,16,32) 
    # assume that the time signature is 4/4
    start_pos =  find_first_nonempty_column(piano_roll[0]["onset_roll"])
    measure_len = resolution # every measure has 4 beats
    start_measure = np.floor(start_pos/measure_len)
    start_measure_beat = int(start_measure*measure_len)
    if start_measure_beat == start_pos: # starting at the beginning of "start measure"
        return start_measure_beat
    else:
        start_measure = start_measure+1
        start_measure_beat = int(start_measure*measure_len)
        return start_measure_beat
    
def find_last_nonempty_measure_beat(piano_roll, resolution=16):
    end_pos = find_last_nonempty_column(piano_roll[0]["sustain_roll"])
    measure_len = resolution
    end_measure = np.floor(end_pos/measure_len)
    end_measure_beat = int(end_measure*measure_len)+measure_len
    if end_measure_beat == end_pos+1:
        return end_measure_beat
    else:
        end_measure = end_measure-1
        end_measure_beat = int(end_measure*measure_len)+measure_len
        return end_measure_beat
    
def cut_piano_roll(piano_roll, resolution=16, lowest=33, highest=96):
    ## we also turn the piano roll into a numpy array
    start_measure_beat = find_first_nonempty_measure_beat(piano_roll, resolution=resolution)
    end_measure_beat = find_last_nonempty_measure_beat(piano_roll, resolution=resolution)
    ##piano_roll_cut=[]
    onset_roll_cut = piano_roll[0]["onset_roll"][lowest:highest+1,start_measure_beat:end_measure_beat]
    sustain_roll_cut = piano_roll[0]["sustain_roll"][lowest:highest+1,start_measure_beat:end_measure_beat]
    #piano_roll_cut.append({"onset_roll": onset_roll_cut, "sustain_roll": sustain_roll_cut})
    piano_roll_cut = np.stack((onset_roll_cut.T,sustain_roll_cut.T))
    return piano_roll_cut


def get_slices(piano_roll, resolution = 16, beats_per_patch=64):
    # piano_roll: (2, L, 64) arrays
    # create slices of one piece
    # for each slice, we store 4 measures, which is 16 beats
    n = int(np.floor(piano_roll.shape[1]/beats_per_patch)) # number of patches that we will get
    slices = []
    for i in range(n):
        start_pos = i*beats_per_patch
        end_pos = (i+1)*beats_per_patch
        slices.append(piano_roll[:,start_pos:end_pos,:])
    slices = np.array(slices)
    return slices



# Now, create the dataset
def create_training_set(root_dir):
    midi_files = glob.glob(os.path.join(root_dir, '*.mid'))
    slices = []
    for midi_file in tqdm.tqdm(midi_files):
        #print(midi_file)
        piano_rolls = midi_to_piano_roll(midi_file)
        piano_roll = combine_piano_roll(piano_rolls)
        piano_roll_cut = cut_piano_roll(piano_roll)
        new_slices = get_slices(piano_roll_cut)
        slices.extend(new_slices)

    return np.array(slices)
        