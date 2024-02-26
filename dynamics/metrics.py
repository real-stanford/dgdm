import torch
import numpy as np

def convergence_mode(profile):
    '''
    return the lengths of sequences in a tensor that are consecutive 1s followed by consecutive 0s, e.g. 1,0/1,1,1,0,0,0/1,1,0,0,0,0. also wrap around the tensor to handle wrapped sequences.
    return the indices of the convergence points, i.e. the indices of the last 1s in the sequences.
    '''
    profile = torch.where(profile > 0, 1.0, 0.0)
    if torch.all(profile == 0):
        return torch.tensor([len(profile)], device=profile.device), torch.tensor([0], device=profile.device)
    elif torch.all(profile == 1):
        return torch.tensor([len(profile)], device=profile.device), torch.tensor([len(profile)-1], device=profile.device)
    profile = torch.cat((profile, profile), dim=0)
    diff = torch.diff(profile)
    convergence_points = torch.where(diff < 0)[0]
    convergence_points = convergence_points[convergence_points < (len(profile) // 2)]
    sequence_start = torch.where(diff > 0)[0]
    sequence_lengths = torch.diff(torch.cat((torch.tensor([0], device=sequence_start.device), sequence_start[sequence_start > convergence_points[0]], torch.tensor([len(profile)], device=sequence_start.device))))
    sequence_lengths = sequence_lengths[:len(convergence_points)]
    return sequence_lengths, convergence_points

def convergence_mode_three_class(profile):
    profile_binary_ids = torch.where(profile != 1)[0]
    if len(profile_binary_ids) == 0:
        return torch.tensor([0], device=profile.device), torch.tensor([0], device=profile.device)
    profile_binary = profile[profile != 1]
    sequence_lengths, convergence_points = convergence_mode(profile_binary)
    convergence_points = profile_binary_ids[convergence_points]
    return sequence_lengths, convergence_points

def slicer(a, lower, upper):
    if lower < 0:
        return torch.cat((a[lower:], a[:upper]))
    elif upper > len(a):
        return torch.cat((a[lower:], a[:upper-len(a)]))
    else:
        return a[lower: upper]

def convergence_range_from_finals(finals, threshold=0.1):
    '''
    given the final orientations, return the convergence range (consecutive range where the finals are close to each other in a threshold)
    finals: [num_ori]
    '''
    convergence_range = []
    start = 0
    end = 0
    min_consecutive_range = 1
    min_range_final = finals[0]
    max_range_final = finals[0]
    for i in range(1, len(finals)):
        min_range_final = min(min_range_final, finals[i])
        max_range_final = max(max_range_final, finals[i])
        if (max_range_final - min_range_final) <= threshold:
            end = i
        else:
            if end - start >= min_consecutive_range:
                convergence_range.append((start, end))
            start = i
            end = i
            min_range_final = finals[i]
            max_range_final = finals[i]
    if end - start >= min_consecutive_range:
        convergence_range.append((start, end))
    return convergence_range

def metric2objective(metric, objective):
    if objective == 'rotate':
        return {
            'success_rate': np.mean((metric['profile'] == 0) | (metric['profile'] == 2), dtype=np.float32),
            'num_zero_classes': np.sum(metric['profile']==1, dtype=np.int16),
            'delta_theta_abs': np.mean(np.abs(metric['delta_theta'])),
            'final_delta_theta_abs': np.mean(np.abs(metric['final_delta_theta'])),
        }
    elif objective == 'rotate_clockwise':
        return{
            'success_rate': np.mean(metric['profile'] == 0, dtype=np.float32),
            'num_clockwise_classes': np.sum(metric['profile']==0, dtype=np.int16),
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
        }
    elif objective == 'rotate_counterclockwise':
        return{
            'success_rate': np.mean(metric['profile'] == 2, dtype=np.float32),
            'num_counterclockwise_classes': np.sum(metric['profile']==2, dtype=np.int16),
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
        }
    elif objective == 'shift_up':   # shift up: x negative
        return {
            'success_rate': np.mean(metric['profile_x'] == 0, dtype=np.float32),
            'num_up_classes': np.sum(metric['profile_x']==0, dtype=np.int16), 
            'delta_pos_x': np.mean(metric['delta_pos'][:, 0]),
            'final_pos_x': np.mean(metric['final_pos'][:, 0]),
        }
    elif objective == 'shift_down': # shift down: x positive
        return {
            'success_rate': np.mean(metric['profile_x'] == 2, dtype=np.float32),
            'num_down_classes': np.sum(metric['profile_x']==2, dtype=np.int16),
            'delta_pos_x': np.mean(metric['delta_pos'][:, 0]),
            'final_pos_x': np.mean(metric['final_pos'][:, 0]),
        }
    elif objective == 'shift_left': # shift left: y negative
        return {
            'success_rate': np.mean(metric['profile_y'] == 0, dtype=np.float32),
            'num_left_classes': np.sum(metric['profile_y']==0, dtype=np.int16),
            'delta_pos_y': np.mean(metric['delta_pos'][:, 1]),
            'final_pos_y': np.mean(metric['final_pos'][:, 1]),
        }
    elif objective == 'shift_right':# shift right: y positive
        return {
            'success_rate': np.mean(metric['profile_y'] == 2, dtype=np.float32),
            'num_right_classes': np.sum(metric['profile_y']==2, dtype=np.int16),
            'delta_pos_y': np.mean(metric['delta_pos'][:, 1]),
            'final_pos_y': np.mean(metric['final_pos'][:, 1]),
        }
    elif objective == 'convergence':
        convergence_range_3deg = convergence_range_from_finals(metric['final_theta'], threshold=3)
        max_convergence_range_3deg = np.max([end - start for start, end in convergence_range_3deg]) if len(convergence_range_3deg) > 0 else 0
        convergence_range_5deg = convergence_range_from_finals(metric['final_theta'], threshold=5)
        max_convergence_range_5deg = np.max([end - start for start, end in convergence_range_5deg]) if len(convergence_range_5deg) > 0 else 0
        convergence_range_10deg = convergence_range_from_finals(metric['final_theta'], threshold=10)
        max_convergence_range_10deg = np.max([end - start for start, end in convergence_range_10deg]) if len(convergence_range_10deg) > 0 else 0
        return {
            'max_convergence_range_3deg': max_convergence_range_3deg,
            'max_convergence_range_5deg': max_convergence_range_5deg,
            'max_convergence_range_10deg': max_convergence_range_10deg,
        }
    elif objective == 'clockwise_up':
        num_clockwise_classes = np.sum(metric['profile']==0, dtype=np.int16)
        num_up_classes = np.sum(metric['profile_x']==0, dtype=np.int16)
        return {
            'success_rate': np.mean((metric['profile'] == 0) & (metric['profile_x'] == 0), dtype=np.float32),
            'num_clockwise_up_classes': num_clockwise_classes + num_up_classes,
            'num_clockwise_classes': num_clockwise_classes,
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
            'num_up_classes': num_up_classes, 
            'delta_pos_x': np.mean(metric['delta_pos'][:, 0]),
            'final_pos_x': np.mean(metric['final_pos'][:, 0]),
        }
    elif objective == 'clockwise_down':
        num_clockwise_classes = np.sum(metric['profile']==0, dtype=np.int16)
        num_down_classes = np.sum(metric['profile_x']==2, dtype=np.int16)
        return {
            'success_rate': np.mean((metric['profile'] == 0) & (metric['profile_x'] == 2), dtype=np.float32),
            'num_clockwise_down_classes': num_clockwise_classes + num_down_classes,
            'num_clockwise_classes': num_clockwise_classes,
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
            'num_down_classes': num_down_classes,
            'delta_pos_x': np.mean(metric['delta_pos'][:, 0]),
            'final_pos_x': np.mean(metric['final_pos'][:, 0]),
        }
    elif objective == 'clockwise_right':
        num_clockwise_classes = np.sum(metric['profile']==0, dtype=np.int16)
        num_right_classes = np.sum(metric['profile_y']==2, dtype=np.int16)
        return {
            'success_rate': np.mean((metric['profile'] == 0) & (metric['profile_y'] == 2), dtype=np.float32),
            'num_clockwise_right_classes': num_clockwise_classes + num_right_classes,
            'num_clockwise_classes': num_clockwise_classes,
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
            'num_right_classes': num_right_classes,
            'delta_pos_y': np.mean(metric['delta_pos'][:, 1]),
            'final_pos_y': np.mean(metric['final_pos'][:, 1]),
        }
    elif objective == 'clockwise_left':
        num_clockwise_classes = np.sum(metric['profile']==0, dtype=np.int16)
        num_left_classes = np.sum(metric['profile_y']==0, dtype=np.int16)
        return {
            'success_rate': np.mean((metric['profile'] == 0) & (metric['profile_y'] == 0), dtype=np.float32),
            'num_clockwise_left_classes': num_clockwise_classes + num_left_classes,
            'num_clockwise_classes': num_clockwise_classes,
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
            'num_left_classes': num_left_classes,
            'delta_pos_y': np.mean(metric['delta_pos'][:, 1]),
            'final_pos_y': np.mean(metric['final_pos'][:, 1]),
        }
    elif objective == 'counterclockwise_up':
        num_counterclockwise_classes = np.sum(metric['profile']==2, dtype=np.int16)
        num_up_classes = np.sum(metric['profile_x']==0, dtype=np.int16)
        return {
            'success_rate': np.mean((metric['profile'] == 2) & (metric['profile_x'] == 0), dtype=np.float32),
            'num_counterclockwise_up_classes': num_counterclockwise_classes + num_up_classes,
            'num_counterclockwise_classes': num_counterclockwise_classes,
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
            'num_up_classes': num_up_classes, 
            'delta_pos_x': np.mean(metric['delta_pos'][:, 0]),
            'final_pos_x': np.mean(metric['final_pos'][:, 0]),
        }
    elif objective == 'counterclockwise_down':
        num_counterclockwise_classes = np.sum(metric['profile']==2, dtype=np.int16)
        num_down_classes = np.sum(metric['profile_x']==2, dtype=np.int16)
        return {
            'success_rate': np.mean((metric['profile'] == 2) & (metric['profile_x'] == 2), dtype=np.float32),
            'num_counterclockwise_down_classes': num_counterclockwise_classes + num_down_classes,
            'num_counterclockwise_classes': num_counterclockwise_classes,
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
            'num_down_classes': num_down_classes,
            'delta_pos_x': np.mean(metric['delta_pos'][:, 0]),
            'final_pos_x': np.mean(metric['final_pos'][:, 0]),
        }
    elif objective == 'counterclockwise_right':
        num_counterclockwise_classes = np.sum(metric['profile']==2, dtype=np.int16)
        num_right_classes = np.sum(metric['profile_y']==2, dtype=np.int16)
        return {
            'success_rate': np.mean((metric['profile'] == 2) & (metric['profile_y'] == 2), dtype=np.float32),
            'num_counterclockwise_right_classes': num_counterclockwise_classes + num_right_classes,
            'num_counterclockwise_classes': num_counterclockwise_classes,
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
            'num_right_classes': num_right_classes,
            'delta_pos_y': np.mean(metric['delta_pos'][:, 1]),
            'final_pos_y': np.mean(metric['final_pos'][:, 1]),
        }
    elif objective == 'counterclockwise_left':
        num_counterclockwise_classes = np.sum(metric['profile']==2, dtype=np.int16)
        num_left_classes = np.sum(metric['profile_y']==0, dtype=np.int16)
        return {
            'success_rate': np.mean((metric['profile'] == 2) & (metric['profile_y'] == 0), dtype=np.float32),
            'num_counterclockwise_left_classes': num_counterclockwise_classes + num_left_classes,
            'num_counterclockwise_classes': num_counterclockwise_classes,
            'delta_theta': np.mean(metric['delta_theta']),
            'final_delta_theta': np.mean(metric['final_delta_theta']),
            'num_left_classes': num_left_classes,
            'delta_pos_y': np.mean(metric['delta_pos'][:, 1]),
            'final_pos_y': np.mean(metric['final_pos'][:, 1]),
        }
    else:
        raise NotImplementedError