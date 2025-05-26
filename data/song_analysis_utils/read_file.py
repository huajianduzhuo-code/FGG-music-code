import numpy as np
import mir_eval
import os
from .song_data_structure import McpaMusic

def create_string(num):
    if num < 8:
        return f"A{num}"
    result = ""
    while num >= 16:
        result += "A8B8"
        num -= 16
    if num > 0:
        if num <= 8:
            result += f"A{num}"
        else:
            result += "A8"
            num -= 8
            result += f"B{num}"
    return result


def _cum_time_to_time_dur(d):
    d_cumsum = np.cumsum(d)
    starts = np.insert(d_cumsum, 0, 0)[0: -1]
    return starts


def _parse_phrase_label(phrase_string):
    phrase_starts = [i for i, s in enumerate(phrase_string) if s.isalpha()] + [len(phrase_string)]
    phrase_names = [phrase_string[phrase_starts[i]: phrase_starts[i + 1]] for i in range(len(phrase_starts) - 1)]
    phrase_types = [pn[0] for pn in phrase_names]
    phrase_lgths = np.array([int(pn[1:]) for pn in phrase_names])
    return phrase_names, phrase_types, phrase_lgths


def read_label(label_fn):
    """label_fn is human_label1.txt or human_label1.txt in the song folder of the dataset."""
    with open(label_fn) as f:
        phrase_label = f.readlines()[0].strip()
    phrase_names, phrase_types, phrase_lgths = _parse_phrase_label(phrase_label)

    phrase_starts = _cum_time_to_time_dur(phrase_lgths)

    phrases = [{'name': pn, 'type': pt, 'lgth': pl, 'start': ps}
               for pn, pt, pl, ps in zip(phrase_names, phrase_types, phrase_lgths, phrase_starts)]

    return phrases


def read_melody(melody_fn):
    """melody_fn is melody.txt in the song folder of the dataset."""

    # convert txt file to numpy array of (pitch, duration)
    with open(melody_fn) as f:
        melody = f.readlines()
    melody = [m.strip().split(' ') for m in melody]
    melody = np.array([[int(m[0]), int(m[1])] for m in melody])

    # convert numpy array of (pitch, duration) to (onset, pitch, dur)
    starts = _cum_time_to_time_dur(melody[:, 1])
    durs = melody[:, 1]
    pitches = melody[:, 0]
    is_pitch = melody[:, 0] != 0
    note_mat = np.stack([starts, pitches, durs], -1)[is_pitch]

    return note_mat


def _read_chord_string(c):
    """
    "\x01"s are replaced with " ". ")"s are added to "sus4(b7".
    (E.g., check with if c_name[-7:] == 'sus4(b7'). And there may have other annotation problems.)
    The files in this repo have been cleaned.
    """
    c = c.strip().split(' ')
    c_name = c[0]
    c_dur = int(c[-1])

    # cleaning the chord symbol
    c_name = c_name.replace('\x01', '')

    # convert to chroma representation
    root, chroma, bass = mir_eval.chord.encode(c_name) # root: root note, chroma: a 12-d list indicating which pitches are included, bass: which one has the lowest pitch
    chroma = np.roll(chroma, shift=root)
    return np.concatenate([np.array([root]), chroma, np.array([bass]), np.array([c_dur])])


def read_chord(chord_fn):
    """chord_fn is finalized_chord.txt in the song folder of the dataset."""
    with open(chord_fn) as f:
        chords = f.readlines()

    # convert chord text label to chroma representation
    chords = np.stack([_read_chord_string(c) for c in chords], 0)

    # convert chord to output chord matrix
    starts = _cum_time_to_time_dur(chords[:, -1])
    chord_mat = np.concatenate([starts[:, np.newaxis], chords], -1)

    return chord_mat


def read_data(data_fn, acc_fn, num_beat_per_measure=4, num_step_per_beat=4,
              clean_chord_unit=None, song_name=None, label=1, track_names_to_keep=['piano']):# delete'bridge'
    melody_fn = os.path.join(data_fn, 'melody.txt')
    melody = read_melody(melody_fn)
    num_measures = int(np.ceil((melody[-1,0]+melody[-1,2]) / num_beat_per_measure / num_step_per_beat))

    if label == 1:
        if not os.path.exists(os.path.join(data_fn, 'human_label1.txt')):
            # Create the file and write "aa" to it
            with open(os.path.join(data_fn, 'human_label1.txt'), 'w') as file:
                # every 8 measure as a part
                file.write(create_string(num_measures))
        label_fn = os.path.join(data_fn, 'human_label1.txt')
    elif label == 2:
        label_fn = os.path.join(data_fn, 'human_label2.txt')
    else:
        raise NotImplementedError

    label = read_label(label_fn)

    chord_fn = os.path.join(data_fn, 'finalized_chord.txt')
    chord = read_chord(chord_fn)

    clean_chord_unit = num_beat_per_measure if clean_chord_unit is None else clean_chord_unit

    # if "drum" in track_names_to_keep:
    #     acc_mats = np.load(os.path.join(acc_fn, 'drum_mat.npz'))
    # else:
    #     acc_mats = np.load(os.path.join(acc_fn, 'acc_mat.npz'))
    
    acc_mats = np.load(os.path.join(acc_fn, 'acc_mat.npz'))

    acc = np.concatenate([acc_mats[key] for key in track_names_to_keep], 0)
    acc = acc[acc[:, 0].argsort()]
    #acc_mats = acc_mats[acc_mats[:, 0].argsort()]

    song = McpaMusic(melody, chord, acc, acc_mats, label, num_beat_per_measure,
                     num_step_per_beat, song_name, clean_chord_unit)

    return song