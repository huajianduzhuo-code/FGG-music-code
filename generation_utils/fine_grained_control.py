import torch
import numpy as np

def get_chord_chroma_mapping(device="cpu"):
    chord_chroma_mapping = [[[1., 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]],
    [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]],
    [[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]],
    [[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]],
    [[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]],
    [[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]],
    [[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]],
    [[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]],
    [[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]],
    [[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]],
    [[1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]],
    [[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]],
    [[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]],
    [[1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]],
    [[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]],
    [[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]],
    [[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]],
    [[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
    [[1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]],
    [[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]],
    [[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]]]

    chord_chroma_mapping = torch.tensor(chord_chroma_mapping, device=device)
    chord_chroma_mapping = torch.tile(chord_chroma_mapping, (1, 1, 64 // chord_chroma_mapping.size(2) + 1))

    rotated_mappings_list = []
    for i in range(chord_chroma_mapping.shape[0]):
        this_map = chord_chroma_mapping[i]
        rotated_this_map = torch.stack([torch.roll(this_map, shifts=-i, dims=1) for i in range(12)], dim=0)[:,:,:64]
        rotated_mappings_list.append(rotated_this_map)
    return torch.concat(rotated_mappings_list, axis=0).to(device)

def edit_rhythm_func(piano_roll_full, num_notes_onset, mask_full):
    '''
    piano_roll_full: a tensor with shape (batch_size, 2, length, h) # length=64 is length of roll, h is number of possible pitch
    num_notes_onset: a tensor with shape (batch_size, length)
    mask_full: a tensor with shape the same as piano_roll
    '''
    ########## for those greater than the threshold, if num of notes exceed num_notes[i], 
    ########## will keep the first ones and set others to threshold
    print("editing rhythm")
    # we only edit onset
    onset_roll = piano_roll_full[:,0,:,:]
    mask = mask_full[:,0,:,:]
    shape = onset_roll.shape

    onset_roll = onset_roll.reshape(-1,shape[-1])
    mask = mask.reshape(-1,shape[-1])
    num_notes = num_notes_onset.reshape(-1)

    reduce_note_threshold = 0.499
    increase_note_threshold = 0.501

    # Initialize a tensor to store the modified values
    final_onset_roll = onset_roll.clone()
    threshold_mask = onset_roll > reduce_note_threshold
    # Set all values <= reduce_note_threshold to -inf to exclude them from top-k selection
    values_above_threshold = torch.where(threshold_mask & (mask == 1), onset_roll, torch.tensor(-float('inf')).to(onset_roll.device))

    # Get the top num_notes.max() values for each row
    num_notes_max = int(num_notes.max().item())  # Maximum number of notes needed in any row
    topk_values, topk_indices = torch.topk(values_above_threshold, num_notes_max, dim=1)

    # Create a mask for the top num_notes[i] values for each row
    col_indices = torch.arange(num_notes_max, device=onset_roll.device).expand(len(onset_roll), num_notes_max)
    topk_mask = (col_indices < num_notes.unsqueeze(1)) & (topk_values > -float("inf"))

    # Set all values greater than reduce_note_threshold to reduce_note_threshold initially
    final_onset_roll[threshold_mask & (mask == 1)] = reduce_note_threshold

    # Create a flattened index to scatter the top values back into final_onset_roll
    flat_row_indices = torch.arange(onset_roll.size(0), device=onset_roll.device).unsqueeze(1).expand_as(topk_indices)
    flat_row_indices = flat_row_indices[topk_mask]

    # Gather the valid topk_indices and corresponding values
    valid_topk_indices = topk_indices[topk_mask]
    valid_topk_values = topk_values[topk_mask]

    # Use scatter to place the top num_notes[i] values back to their original positions
    final_onset_roll = final_onset_roll.index_put_((flat_row_indices, valid_topk_indices), valid_topk_values)


    # Count how many values >= increase_note_threshold for each row
    threshold_mask_2 = (final_onset_roll >= increase_note_threshold)&(mask==1)
    greater_than_threshold2_count = threshold_mask_2.sum(dim=1)

    # For those rows, find the remaining number of values needed to be set to increase_note_threshold
    remaining_needed = num_notes - greater_than_threshold2_count
    remaining_needed_max = int(remaining_needed.max().item())

    # Find the values in each row that are < increase_note_threshold but are the highest (so we can set them to increase_note_threshold)
    values_below_threshold2 = torch.where((final_onset_roll < increase_note_threshold)&(mask==1), final_onset_roll, torch.tensor(-float('inf')).to(onset_roll.device))
    topk_below_threshold2_values, topk_below_threshold2_indices = torch.topk(values_below_threshold2, remaining_needed_max, dim=1)

    # Mask to only adjust the needed number of values in each row
    col_indices_below_threshold2 = torch.arange(remaining_needed_max, device=onset_roll.device).expand(len(onset_roll), remaining_needed_max)
    adjust_mask = (col_indices_below_threshold2 < remaining_needed.unsqueeze(1)) & (topk_below_threshold2_values > -float("inf"))

    # Flatten row indices for the new top-k below increase_note_threshold
    flat_row_indices_below_threshold2 = torch.arange(onset_roll.size(0), device=onset_roll.device).unsqueeze(1).expand_as(topk_below_threshold2_indices)
    flat_row_indices_below_threshold2 = flat_row_indices_below_threshold2[adjust_mask]

    # Gather the valid indices and set them to increase_note_threshold
    valid_below_threshold2_indices = topk_below_threshold2_indices[adjust_mask]

    # Update the final_onset_roll to make sure we now have exactly num_notes[i] values >= increase_note_threshold
    final_onset_roll = final_onset_roll.index_put_((flat_row_indices_below_threshold2, valid_below_threshold2_indices), torch.tensor(increase_note_threshold, device=onset_roll.device))
    final_onset_roll = final_onset_roll.reshape(shape)
    piano_roll_full[:,0,:,:] = final_onset_roll
    return piano_roll_full

def X0EditFunc(x0, background_condition, sampler_device="cpu", edit_rhythm=True):
    print("editing")
    chd_scale_map = get_chord_chroma_mapping(device=sampler_device)
    # if using null rhythm condition, have to convert -2 to 1 and -1 to 0
    if background_condition[:,:2,:,:].min()<0:
        correct_chord_condition = -background_condition[:,:2,:,:]-1
    else:
        correct_chord_condition = background_condition[:,:2,:,:]
    merged_chd_roll = torch.max(correct_chord_condition[:,0,:,:], correct_chord_condition[:,1,:,:]) # chd roll of our bg_cond
    chd_chroma_ours = torch.clamp(merged_chd_roll, min=0.0, max=1.0) # chd chroma of our bg_cond
    shape = chd_chroma_ours.shape
    chd_chroma_ours = chd_chroma_ours.reshape(-1,64)
    matches = (chd_scale_map[:, 0, :].unsqueeze(0)[:,:,15:39] == chd_chroma_ours.unsqueeze(1)[:,:,15:39]).all(dim=-1)

    excess_match_positions = matches.cumsum(dim=-1) > 1
    matches[excess_match_positions] = False
    assert matches.sum(-1).max()<=1
    seven_notes_chroma_ours = torch.einsum('ij,jk->ik', matches.float(), chd_scale_map[:, 1, :]).reshape(shape)
    seven_notes_chroma_ours = seven_notes_chroma_ours.unsqueeze(1).repeat((1,2,1,1))

    no_chd_match = torch.all(seven_notes_chroma_ours == 0, dim=-1)
    seven_notes_chroma_ours[no_chd_match] = 1.
    
    # edit notes based on chroma
    x0 = torch.where((seven_notes_chroma_ours==0)&(x0>0.5), 0.5 , x0)
    
    if edit_rhythm:
        # edit rhythm
        if background_condition[:,:2,:,:].min()>=0: # only edit if rhythm is provided
            num_onset_notes, _ = torch.max(background_condition[:,0,:,:], axis=-1)
            x0 = edit_rhythm_func(x0, num_onset_notes, seven_notes_chroma_ours)

    return x0

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