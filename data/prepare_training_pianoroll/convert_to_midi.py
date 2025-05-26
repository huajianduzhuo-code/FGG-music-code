'''
Utils for converting piano_rolls into midi files
'''


import pretty_midi as pm
import numpy as np

def default_quantization(v):
    return 1 if v > 0.5 else 0

def extend_piano_roll(piano_roll: np.ndarray, lowest=33, highest=96):
    ## this function is for extending the cutted piano rolls into the full 128 piano rolls
    ## recall that the piano rolls are of dimensions (2,L,64), we add zeros and fill it into (2,L,128)
    padded_roll = np.pad(piano_roll, ((0, 0), (0, 0), (lowest, 127-highest)), mode='constant', constant_values=0)
    return padded_roll



def piano_roll_to_note_mat(piano_roll: np.ndarray, quantization_func=default_quantization):
    """
    piano_roll: (2, L, 128), onset and sustain channel.
    raise_chord: whether pitch below 48 (mel-chd boundary) will be raised an octave
    """
    def convert_p(p_, note_list):
        edit_note_flag = False
        for t in range(n_step):
            onset_state = quantization_func(piano_roll[0, t, p_])
            sustain_state = quantization_func(piano_roll[1, t, p_])

            is_onset = bool(onset_state)
            is_sustain = bool(sustain_state) and not is_onset

            pitch = p_ 

            if is_onset:
                edit_note_flag = True
                note_list.append([t, pitch, 1])
            elif is_sustain:
                if edit_note_flag:
                    note_list[-1][-1] += 1
            else:
                edit_note_flag = False
        return note_list

    quantization_func = default_quantization if quantization_func is None else quantization_func
    assert len(piano_roll.shape) == 3 and piano_roll.shape[0] == 2 and piano_roll.shape[2] == 128, f"{piano_roll.shape}" 

    n_step = piano_roll.shape[1]

    notes = []
    for p in range(128):
        convert_p(p, notes)

    return notes


def note_mat_to_notes(note_mat, bpm, unit=1/4, shift_beat=0., shift_sec=0., vel=100):
    """Default use shift beat"""

    beat_alpha = 60 / bpm
    step_alpha = unit * beat_alpha

    notes = []

    shift_sec = shift_sec if shift_beat is None else shift_beat * beat_alpha

    for note in note_mat:
        onset, pitch, dur = note
        start = onset * step_alpha + shift_sec
        end = (onset + dur) * step_alpha + shift_sec

        notes.append(pm.Note(vel, int(pitch), start, end))

    return notes


def create_pm_object(bpm, piano_notes_list, chd_notes_list, melody_notes_list=None):
    midi = pm.PrettyMIDI(initial_tempo=bpm)

    if piano_notes_list is not None:
        piano_program = pm.instrument_name_to_program('Acoustic Grand Piano')
        piano = pm.Instrument(program=piano_program)
        piano.notes+=piano_notes_list
        midi.instruments.append(piano)

    if chd_notes_list is not None:
        chd_program = pm.instrument_name_to_program('Acoustic Guitar (steel)')
        chd = pm.Instrument(program=chd_program)
        chd.notes+=chd_notes_list
        midi.instruments.append(chd)

    if melody_notes_list is not None:
        melody_program = pm.instrument_name_to_program('Acoustic Grand Piano')
        melody = pm.Instrument(program=melody_program)
        melody.notes+=melody_notes_list
        midi.instruments.append(melody)

    return midi

def piano_roll_to_midi(piano_roll, chd_roll, melody_roll = None, bpm=100):
    if piano_roll is not None:
        piano_mat = piano_roll_to_note_mat(piano_roll)
        piano_notes = note_mat_to_notes(piano_mat, bpm)
    else:
        piano_notes = None

    if chd_roll is not None:
        chd_mat = piano_roll_to_note_mat(chd_roll)
        chd_notes = note_mat_to_notes(chd_mat, bpm, vel=60)
    else:
        chd_notes=None

    if melody_roll is not None:
        melody_mat = piano_roll_to_note_mat(melody_roll)
        melody_notes = note_mat_to_notes(melody_mat, bpm)
    else:
        melody_notes=None

    piano_pm = create_pm_object(bpm = 80, piano_notes_list=piano_notes, 
                                chd_notes_list=chd_notes, melody_notes_list=melody_notes)
    return piano_pm

def save_midi(pm, filename):
    pm.write(filename)