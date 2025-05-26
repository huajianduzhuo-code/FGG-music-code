import numpy as np
import os
from tqdm import tqdm
import pickle
import argparse

from prepare_training_pianoroll.get_piano_roll import get_slices
from song_analysis_utils.format_converter import note_matrix_to_piano_roll, chord_mat_to_chord_roll
from song_analysis_utils.read_pop909_data import read_pop909_dataset
from song_analysis_utils.read_file import McpaMusic

SPLIT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'split.npz')

def load_split_file(split_fn):
    split_data = np.load(split_fn)
    train_inds = split_data['train_inds']
    valid_inds = split_data['valid_inds']
    return train_inds, valid_inds

def expand_roll(roll, unit=4, contain_onset=False):
    # roll: (Channel, T, H) -> (Channel, T * unit, H)
    n_channel, length, height = roll.shape

    expanded_roll = roll.repeat(unit, axis=1)
    if contain_onset:
        expanded_roll = expanded_roll.reshape((n_channel, length, unit, height))
        expanded_roll[1::2, :, 1:] = np.maximum(expanded_roll[::2, :, 1:], expanded_roll[1::2, :, 1:])

        expanded_roll[::2, :, 1:] = 0
        expanded_roll = expanded_roll.reshape((n_channel, length * unit, height))
    return expanded_roll

def cut_piano_roll(piano_roll, resolution=16, lowest=33, highest=96):
    piano_roll_cut = piano_roll[:,:,lowest:highest+1]
    return piano_roll_cut

def circular_extend(chd_roll, lowest=33, highest=96):
    #chd_roll: 6*L*12->6*L*64
    C4 = 60-lowest
    C3 = C4-12
    shape = chd_roll.shape
    ext_chd = np.zeros((shape[0],shape[1],highest+1-lowest))
    ext_chd[:,:,C4:C4+12] = chd_roll
    ext_chd[:,:,C3:C3+12] = chd_roll
    return ext_chd

def data_preprocess(dataset, augmentation=True):
    """
    Process musical data to create training/test slices for chord training.
    
    This function processes a dataset of musical pieces to create piano roll slices containing
    accompaniment, chord, and melody information. It handles data augmentation through pitch shifting
    when enabled. The function processes each piece to extract:
    - Accompaniment rolls (onset and sustain)
    - Chord rolls (onset with rhythm, sustain with rhythm, onset without rhythm, sustain without rhythm)
    - Melody rolls (onset and sustain)
    
    Args:
        dataset: List of musical pieces to process
        augmentation (bool): Whether to perform data augmentation through pitch shifting.
                           When True, creates additional samples by shifting pitches up and down.
    
    Returns:
        numpy.ndarray: Array of shape (N, 8, L, H) containing all processed slices where:
            - N: Number of samples
            - 8: Number of channels (2 accompaniment + 4 chord + 2 melody)
            - L: Length of piano rolls
            - H: Number of note pitches
    """
    acc_slices = []
    chd_slices = []
    melody_slices = []

    for i, file in enumerate(dataset):
        # extract the rolls: acc and chord
        extract = ComponentsExtractor(dataset[i])
        acc_roll = extract.extract_accompaniment()
        melody_roll = extract.extract_lead_sheet()

        # expand the chord roll (times 4)
        chd_roll = expand_roll(melody_roll['chd_roll'], contain_onset=True)
        mel_roll = melody_roll['mel_roll']

        # number of onset notes and sustain notes on each position
        acc_num_onset_notes = acc_roll["acc_roll"][0].sum(axis=-1)
        chd_roll_mixed = np.max(chd_roll[2:4,:,:], axis=0)

        # chord roll is no longer 0,1 valued, the value is number of onset notes
        chd_roll_onset = chd_roll_mixed*acc_num_onset_notes[:, np.newaxis]
        # sustain roll is empty if there exists onset notes, otherwise a 0,1 array corresponding to the current chord
        chd_roll_sustain = np.zeros_like(chd_roll_mixed)
        no_onset_pos = np.all(chd_roll_onset == 0, axis=-1)
        chd_roll_sustain[no_onset_pos] = chd_roll_mixed[no_onset_pos]
        chd_roll = np.concatenate([chd_roll_onset[np.newaxis,:,:], chd_roll_sustain[np.newaxis,:,:], chd_roll[2:4,:,:]], axis=0)
        
        chd_roll_ext_full = circular_extend(chd_roll)
        chd_roll_ext = chd_roll_ext_full
        
        chd_roll_ext[2] = -np.max(chd_roll_ext[2:4,:,:], axis=0)-1
        chd_roll_ext[3] = chd_roll_ext[2]

        # cut accompaniment rolls
        acc_roll_cut = cut_piano_roll(acc_roll['acc_roll'])
        mel_roll_cut = cut_piano_roll(mel_roll)

        # get slices
        new_slices_chd = get_slices(chd_roll_ext)
        new_slices_acc = get_slices(acc_roll_cut)
        new_slices_mel = get_slices(mel_roll_cut)

        chd_slices.extend(new_slices_chd)
        acc_slices.extend(new_slices_acc)
        melody_slices.extend(new_slices_mel)

        if augmentation:
            # shift up
            for shift in range(1,7):
                C4 = 60-33
                C3 = C4-12
                new_slices_chd_shifted = np.zeros_like(new_slices_chd)
                new_slices_chd_12_len = new_slices_chd[:,:,:,27:39] # 先把chord改回长度12的array，便于处理
                new_slices_chd_12_len = np.concatenate([new_slices_chd_12_len[:,:,:,-shift:],new_slices_chd_12_len[:,:,:,:-shift]],axis=-1)
                new_slices_chd_shifted[:,:,:,C4:C4+12] = new_slices_chd_12_len
                new_slices_chd_shifted[:,:,:,C3:C3+12] = new_slices_chd_12_len

                new_slices_acc_shifted = np.concatenate([np.zeros_like(new_slices_acc[:,:,:,-shift:]),new_slices_acc[:,:,:,:-shift]],axis=-1)
                new_slices_mel_shifted = np.concatenate([np.zeros_like(new_slices_mel[:,:,:,-shift:]),new_slices_mel[:,:,:,:-shift]],axis=-1)

                chd_slices.extend(new_slices_chd_shifted)
                acc_slices.extend(new_slices_acc_shifted)
                melody_slices.extend(new_slices_mel_shifted)
            # shift down
            for shift in range(1,6):
                C4 = 60-33
                C3 = C4-12
                new_slices_chd_shifted = np.zeros_like(new_slices_chd)
                new_slices_chd_12_len = new_slices_chd[:,:,:,27:39] # 先把chord改回长度12的array，便于处理
                new_slices_chd_12_len = np.concatenate([new_slices_chd_12_len[:,:,:,shift:],new_slices_chd_12_len[:,:,:,:shift]],axis=-1)
                new_slices_chd_shifted[:,:,:,C4:C4+12] = new_slices_chd_12_len
                new_slices_chd_shifted[:,:,:,C3:C3+12] = new_slices_chd_12_len

                new_slices_acc_shifted = np.concatenate([new_slices_acc[:,:,:,shift:], np.zeros_like(new_slices_acc[:,:,:,:shift])],axis=-1)
                new_slices_mel_shifted = np.concatenate([new_slices_mel[:,:,:,shift:],np.zeros_like(new_slices_mel[:,:,:,:shift])],axis=-1)

                chd_slices.extend(new_slices_chd_shifted)
                acc_slices.extend(new_slices_acc_shifted)
                melody_slices.extend(new_slices_mel_shifted)
    
    chd_slices = np.array(chd_slices)
    acc_slices = np.array(acc_slices)
    melody_slices = np.array(melody_slices)

    slices_all = np.concatenate((acc_slices, chd_slices, melody_slices), axis = 1)
    return slices_all

class ComponentsExtractor:

    def __init__(self, song: McpaMusic):
        self.song = song

        self._mel_roll = None
        self._chd_roll = None
        self._song_dict = None

    def extract_lead_sheet(self):
        mel_roll = note_matrix_to_piano_roll(self.song.melody, self.song.total_step)
        chd_roll = chord_mat_to_chord_roll(self.song.chord, self.song.total_beat)

        self._mel_roll = mel_roll
        self._chd_roll = chd_roll

        return {'mel_roll': mel_roll, 'chd_roll': chd_roll}

    def extract_accompaniment(self):
        acc_roll = note_matrix_to_piano_roll(self.song.acc, self.song.total_step)
        return {'acc_roll': acc_roll}


if __name__ == "__main__":
    '''
    generates arrays with shape (N, C, L, H), where N is sample size, C is number of channels,
    L is length of piano rolls, H is number of note pitches.

    Channels meaning: 

    - if combine_melody_acc is True, C=6:
        [accompaniment onset + melody onset, accompaniment sustain + melody sustain,
     chord onset with rhythm, chord sustain with rhythm,
     chord onset with null rhythm, chord sustain with null rhythm]

    - if combine_melody_acc is False, C=8:
        [accompaniment onset, accompaniment sustain, chord onset with rhythm, chord sustain with rhythm,
     chord onset with null rhythm, chord sustain with null rhythm, melody onset, melody sustain]
    '''
    parser = argparse.ArgumentParser(description='Generate training and test data for chord trainer')
    parser.add_argument('--split_file', type=str, default=SPLIT_FILE_PATH,
                      help='Path to the split file (default: data/split.npz)')
    parser.add_argument('--combine_melody_acc', action='store_true',
                      help='add this flag if want to combine melody and accompaniment in the same channels so that we are training a model that generates melody and accompaniment conditioning on chord and null rhythm; if not added, we are training a model that generates accompaniment conditioning on chord and melody')
    parser.add_argument('--augmentation', type=bool, default=True,
                      help='Augmentation')
    args = parser.parse_args()

    # load dataset
    train_ids, valid_ids = load_split_file(args.split_file)
    train_dataset = read_pop909_dataset(train_ids + 1, desc_dataset='train set')
    test_dataset = read_pop909_dataset(valid_ids + 1, desc_dataset='valid set')

    # preprocess
    train_slices_all = data_preprocess(train_dataset, augmentation=args.augmentation)
    test_slices_all = data_preprocess(test_dataset, augmentation=False)

    if args.combine_melody_acc:
        # Combine accompaniment and leadsheet dimensions
        # Original shape: (N, 8, L, H) -> New shape: (N, 6, L, H)
        # Combine acc_onset + mel_onset and acc_sustain + mel_sustain
        train_slices_all[:,:2,:,:] = np.maximum(train_slices_all[:,-2:,:,:], train_slices_all[:,:2,:,:])
        train_slices_all = train_slices_all[:,:-2,:,:]
        
        test_slices_all[:,:2,:,:] = np.maximum(test_slices_all[:,-2:,:,:], test_slices_all[:,:2,:,:])
        test_slices_all = test_slices_all[:,:-2,:,:]

        # save data
        with open(os.path.join(os.path.dirname(__file__), 'train_test_slices/train_slices_combine_melody_accompaniment.pkl'), 'wb') as f:
            pickle.dump(train_slices_all, f)

        with open(os.path.join(os.path.dirname(__file__), 'train_test_slices/test_slices_combine_melody_accompaniment.pkl'), 'wb') as f:
            pickle.dump(test_slices_all, f)
    else:
        with open(os.path.join(os.path.dirname(__file__), 'train_test_slices/train_slices_separate_melody_accompaniment.pkl'), 'wb') as f:
            pickle.dump(train_slices_all, f)

        with open(os.path.join(os.path.dirname(__file__), 'train_test_slices/test_slices_separate_melody_accompaniment.pkl'), 'wb') as f:
            pickle.dump(test_slices_all, f)