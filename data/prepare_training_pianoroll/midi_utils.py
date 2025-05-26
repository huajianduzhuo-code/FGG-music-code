
import pretty_midi

PIANO = (0,1)
STRING = (40,41,42,48,49,50,51)
GUITAR = (24,25,27)
BRASS = (56,57,58,59,61,62,63,64,65,66,67)

############################## For single track analysis

def calculate_active_duration(instrument):
    # Collect all note intervals
    intervals = [(note.start, note.end) for note in instrument.notes]
    
    # Sort intervals by start time
    intervals.sort()
    
    # Merge overlapping intervals and calculate the active duration
    active_duration = 0
    current_start, current_end = intervals[0]
    
    for start, end in intervals[1:]:
        if start <= current_end:  # There is an overlap
            current_end = max(current_end, end)
        else:  # No overlap, add the previous interval duration and start a new interval
            active_duration += current_end - current_start
            current_start, current_end = start, end
    
    # Add the last interval
    active_duration += current_end - current_start

    return active_duration

def is_full_track(midi, instrument, threshold=0.6):
    # Calculate the total duration of the track
    total_duration = midi.get_end_time()

    # Calculate the active duration (time during which notes are playing)
    active_duration = calculate_active_duration(instrument)

    # Calculate the percentage of active duration
    active_percentage = active_duration / total_duration

    #print(f"Total duration: {total_duration:.2f} seconds")
    #print(f"Active duration: {active_duration:.2f} seconds")
    #print(f"Active percentage: {active_percentage:.2%}")

    # Check if the active duration meets or exceeds the threshold
    return active_percentage >= threshold

#################################### For gathering full tracks

def gather_instr(pm):
    # Gather all the program indexes of the instrument tracks
    program_indexes = [instrument.program for instrument in pm.instruments]
    
    # Sort the program indexes
    program_indexes.sort()
    
    # Convert the sorted list of program indexes to a tuple
    program_indexes_tuple = tuple(program_indexes)
    return program_indexes_tuple

def gather_full_instr(pm, threshold = 0.6):
    # Gather all the program indexes of the instrument tracks that exceed the duration threshold
    program_indexes = []
    for instrument in pm.instruments:
        if is_full_track(pm, instrument, threshold):
            program_indexes.append(instrument.program)
    program_indexes.sort()
    # Convert the list of program indexes to a tuple
    program_indexes_tuple = tuple(program_indexes)
    
    return program_indexes_tuple

####################################### For finding instruments

def has_intersection(wanted_instr, exist_instr):
    # Convert both the tuple and the group of integers to sets
    tuple_set = set(wanted_instr)
    group_set = set(exist_instr)
    
    # Check if there is any intersection
    return not tuple_set.isdisjoint(group_set)

# The functions checking instruments in the midi file tracks
def has_piano(exist_instr):
    wanted_instr = PIANO
    return has_intersection(wanted_instr, exist_instr)

def has_string(exist_instr):
    wanted_instr = STRING
    return has_intersection(wanted_instr, exist_instr)

def has_guitar(exist_instr):
    wanted_instr = GUITAR
    return has_intersection(wanted_instr, exist_instr)

def has_brass(exist_instr):
    wanted_instr = BRASS
    return has_intersection(wanted_instr, exist_instr)

def has_drums(pm):
    for instrument in pm.instruments:
        if instrument.is_drum:
            return True
    return False

    
def print_track_details(instrument):
    """
    For visualizing the information in a midi track
    """
    print(f"Instrument: {pretty_midi.program_to_instrument_name(instrument.program)}")
    print(f"Is drum: {instrument.is_drum}")
    
    print("\nNotes:")
    for note in instrument.notes:
        print(f"Start: {note.start:.2f}, End: {note.end:.2f}, Pitch: {note.pitch}, Velocity: {note.velocity}")

    print("\nControl Changes:")
    for cc in instrument.control_changes:
        print(f"Time: {cc.time:.2f}, Number: {cc.number}, Value: {cc.value}")

    print("\nPitch Bends:")
    for pb in instrument.pitch_bends:
        print(f"Time: {pb.time:.2f}, Pitch: {pb.pitch}")

def is_timesig_44(pm):
    for time_signature in pm.time_signature_changes:
        if time_signature.numerator != 4 or time_signature.denominator != 4:
            return False
    return True

def is_timesig_34(pm):
    for time_signature in pm.time_signature_changes:
        if time_signature.numerator != 4 or time_signature.denominator != 4:
            return False
    return True