import numpy as np
import torch
from multiprocessing import Pool


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param padding if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """
    # xs = xs - 1
    # ys = ys - 1
    
    xs_mask = (xs >= sensor_size[1]) + (xs < 0) 
    ys_mask = (ys >= sensor_size[0]) + (ys < 0) 
    mask = xs_mask + ys_mask
    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0
    
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = torch.zeros(img_size).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps, accumulate=True)
    return img


def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search sorted pytorch tensor
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        if t[l] == x:
            return l
        if t[r] == x:
            return r
            
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device

    if ts.sum() == 0 or len(ts) <= 3:
        return torch.zeros([B, sensor_size[0], sensor_size[1]], device=device)

    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0] + 1e-6
    delta_t = dt / B
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = ts[0] + delta_t*bi
            tend = tstart + delta_t
            beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
            end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins


def events_to_stack_polarity(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240)):
    """
    xs: torch.tensor, [xs]
    ys: torch.tensor, [ys]
    ts: torch.tensor, [ts]
    ps: torch.tensor, [ps]
    B: int, number of bins in output voxel grids 
    device: str or torch.device, device to put voxel grid. If left empty, same device as events
    sensor_size: tuple, the size of the event sensor/output voxels

    Returns: stack: torch.tensor, 2xBxHxW, stack of the events between t0 and t1, 2 refers to two polarities
    """
    if device is None:
        device = xs.device

    if ts.sum() == 0 or len(ts) <= 3:
        return torch.zeros([B, sensor_size[0], sensor_size[1]], device=device)

    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    positives = []
    negtives = []
    dt = ts[-1]-ts[0] + 1e-6
    delta_t = dt / B
    for bi in range(B):
        tstart = ts[0] + delta_t*bi
        tend = tstart + delta_t
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1

        mask_pos = ps[beg:end].clone()
        mask_neg = ps[beg:end].clone()
        mask_pos[ps[beg:end] < 0] = 0
        mask_neg[ps[beg:end] > 0] = 0

        vp = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end] * mask_pos, device, sensor_size=sensor_size,
                clip_out_of_range=False)
        vn = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end] * mask_neg, device, sensor_size=sensor_size,
                clip_out_of_range=False)

        positives.append(vp)
        negtives.append(vn)

    positives_b = torch.stack(positives)
    negtives_b = torch.stack(negtives)
    stack = torch.stack([positives_b, negtives_b])

    return stack


def events_to_stack_no_polarity(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240)):
    """
    xs: torch.tensor, [xs]
    ys: torch.tensor, [ys]
    ts: torch.tensor, [ts]
    ps: torch.tensor, [ps]
    B: int, number of bins in output voxel grids 
    device: str or torch.device, device to put voxel grid. If left empty, same device as events
    sensor_size: tuple, the size of the event sensor/output voxels

    Returns: stack: torch.tensor, BxHxW, stack of the events between t0 and t1
    """
    if device is None:
        device = xs.device

    if ts.sum() == 0 or len(ts) <= 3:
        return torch.zeros([B, sensor_size[0], sensor_size[1]], device=device)

    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0] + 1e-6
    delta_t = dt / B
    for bi in range(B):
        tstart = ts[0] + delta_t*bi
        tend = tstart + delta_t
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1

        b = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end], device, sensor_size=sensor_size,
                clip_out_of_range=False)

        bins.append(b)

    stack = torch.stack(bins)

    return stack

# **************************************
def events_to_image(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into an image.
    xs, ys, ps: torch.tensor, [N]
    """
    # xs = xs - 1
    # ys = ys - 1

    xs_mask = (xs >= sensor_size[1]) + (xs < 0) 
    ys_mask = (ys >= sensor_size[0]) + (ys < 0) 
    mask = xs_mask + ys_mask
    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)

    ys = sensor_size[0]-ys-1

    img.index_put_((ys, xs), ps, accumulate=True)

    return img


def events_to_voxel(xs, ys, ts, ps, num_bins, sensor_size=(180, 240)):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """

    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)

    voxel = []
    ts = ts * (num_bins - 1)
    zeros = torch.zeros(ts.size())
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        voxel.append(voxel_bin)

    return torch.stack(voxel)


def events_to_channels(xs, ys, ps, sensor_size=(180, 240)):
    """
    Generate a two-channel event image containing event counters.
    """

    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])


def events_to_mask(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into a binary mask.
    """
    # xs = xs - 1
    # ys = ys - 1

    xs_mask = (xs >= sensor_size[1]) + (xs < 0) 
    ys_mask = (ys >= sensor_size[0]) + (ys < 0) 
    mask = xs_mask + ys_mask
    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0

    device = xs.device
    img_size = list(sensor_size)
    mask = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    mask.index_put_((ys, xs), ps.abs(), accumulate=False)

    return mask


def events_polarity_mask(ps):
    """
    Creates a two channel tensor that acts as a mask for the input event list.
    :param ps: [N] tensor with event polarity ([-1, 1])
    :return [N x 2] event representation
    """
    inp_pol_mask = torch.stack([ps, ps])
    inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
    inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
    inp_pol_mask[1, :] *= -1

    return inp_pol_mask.transpose(0, 1)


def get_hot_event_mask(event_rate, idx, max_px=100, min_obvs=5, max_rate=0.8):
    """
    Returns binary mask to remove events from hot pixels.
    """

    mask = torch.ones(event_rate.shape).to(event_rate.device)
    if idx > min_obvs:
        for i in range(max_px):
            argmax = torch.argmax(event_rate)
            index = (argmax // event_rate.shape[1], argmax % event_rate.shape[1])
            if event_rate[index] > max_rate:
                event_rate[index] = 0
                mask[index] = 0
            else:
                break
    return mask


def python_event_redistribute_PolarityStack(event_stack, mode='linear'):
    """
    event_stack: torch.tensor, [B, P, C, Y, X], P refers to polarities(i.e. 2)
    mode: int, method to assign event timestamp using linear or random
    return: torch.tensor, batched event cloud: [B, max_num_event, 4] [x, y, t, p]
    """
    batch = event_stack.size()[0]
    num_bins = event_stack.size()[2]

    event_stack = event_stack.round()
    event_cloud = []
    maxlen = 0
    batched_event_cloud = torch.zeros([batch, 1, 4])

    if event_stack.sum() != 0:
        for entry in event_stack:
            if entry.sum() != 0:
                elist = []
                ecoors = torch.nonzero(entry) # N x 4, [p, c, y, x] 
                for ecoor in ecoors:
                    value = entry[ecoor[0], ecoor[1], ecoor[2], ecoor[3]]
                    num_event = int(torch.abs(value).item())
                    el = torch.zeros([num_event, 4]) # [x, y, t, p]
                    el[:, 0] = torch.full([num_event], ecoor[3])
                    el[:, 1] = torch.full([num_event], ecoor[2])
                    t0 = ecoor[1]/num_bins + 1/(100*num_bins)
                    t1 = (ecoor[1]+1)/num_bins
                    el[:, 2] = torch.linspace(t0, t1, num_event) if mode == 'linear' else torch.rand([num_event]) * (t1-t0) + t0
                    el[:, 3] = torch.full([num_event], 1 if value > 0 else -1)
                    elist.append(el)
                elist = torch.cat(elist, dim=0)
                elist = sorted(elist, key=lambda x: x[2])
                elist = torch.stack(elist, dim=0)
            else:
                elist = torch.zeros([1, 4])

            event_cloud.append(elist)

        for entry in event_cloud:
            maxlen = entry.size(0) if entry.size(0) > maxlen else maxlen

        batched_event_cloud = torch.zeros((len(event_cloud), maxlen, 4))

        for batch_idx in range(len(event_cloud)):
            lens = event_cloud[batch_idx].size(0)
            batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

    return batched_event_cloud


def python_event_redistribute_NoPolarityStack(event_stack, mode='linear'):
    """
    event_stack: torch.tensor, [B, C, Y, X]
    mode: int, method to assign event timestamp using linear or random
    return: torch.tensor, batched event cloud: [B, max_num_event, 4] [x, y, t, p]
    """
    batch = event_stack.size()[0]
    num_bins = event_stack.size()[1]

    event_stack = event_stack.round()
    event_cloud = []
    maxlen = 0
    batched_event_cloud = torch.zeros([batch, 1, 4])

    if event_stack.sum() != 0:
        for entry in event_stack:
            if entry.sum() != 0:
                elist = []
                ecoors = torch.nonzero(entry) # N x 3, [c, y, x] 
                for ecoor in ecoors:
                    value = entry[ecoor[0], ecoor[1], ecoor[2]]
                    num_event = int(torch.abs(value).item())
                    el = torch.zeros([num_event, 4]) # [x, y, t, p]
                    el[:, 0] = torch.full([num_event], ecoor[2])
                    el[:, 1] = torch.full([num_event], ecoor[1])
                    t0 = ecoor[0]/num_bins + 1/(100*num_bins)
                    t1 = (ecoor[0]+1)/num_bins
                    el[:, 2] = torch.linspace(t0, t1, num_event) if mode == 'linear' else torch.rand([num_event]) * (t1-t0) + t0
                    el[:, 3] = torch.full([num_event], 1 if value > 0 else -1)
                    elist.append(el)
                elist = torch.cat(elist, dim=0)
                elist = sorted(elist, key=lambda x: x[2])
                elist = torch.stack(elist, dim=0)
            else:
                elist = torch.zeros([1, 4])

            event_cloud.append(elist)

        for entry in event_cloud:
            maxlen = entry.size(0) if entry.size(0) > maxlen else maxlen

        batched_event_cloud = torch.zeros((len(event_cloud), maxlen, 4))

        for batch_idx in range(len(event_cloud)):
            lens = event_cloud[batch_idx].size(0)
            batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

    return batched_event_cloud


def cython_event_redistribute(event_stack, mode='linear'):
    event_stack = event_stack.cpu().numpy().astype(np.float32)
    if mode == 'linear':
        cmode = 0
    elif mode == 'random':
        cmode = 1
    else:
        raise Exception(f'Not support {mode}')

    if len(event_stack.shape) == 5:
        event_cloud = c_event_redistribute.event_redistribute_PolarityStack(event_stack, cmode).astype(np.float32)
    elif len(event_stack.shape) == 4:
        event_cloud = c_event_redistribute.event_redistribute_NoPolarityStack(event_stack, cmode).astype(np.float32)
    else: 
        raise Exception('wrong event stack')

    event_cloud = torch.from_numpy(event_cloud)

    return event_cloud


def func_wrapper_PolarityStack(args):
        return c_event_redistribute.event_redistribute_PolarityStack(*args)


def func_wrapper_NoPolarityStack(args):
        return c_event_redistribute.event_redistribute_NoPolarityStack(*args)


def multiprocess_cython(event_stack, mode='linear'):
    """
    params: event_stack: np.ndarray, B x P x C x H x W
    params: mode: int, 0 for 'linear' and 1 for 'random'
    return: event_cloud: np.ndarray, B x N x 4, [x, y, t, p]
    """
    event_stack = event_stack.cpu().numpy().astype(np.float32)
    if mode == 'linear':
        cmode = 0
    elif mode == 'random':
        cmode = 1
    else:
        raise Exception(f'Not support {mode}')

    batch = event_stack.shape[0]
    maxlen = 0

    with Pool() as p:
        if len(event_stack.shape) == 4:
            event_cloud = p.map(func_wrapper_NoPolarityStack, 
                            [[i[np.newaxis, ...], cmode] for i in event_stack])
        elif len(event_stack.shape) == 5:
            event_cloud = p.map(func_wrapper_PolarityStack, 
                            [[i[np.newaxis, ...], cmode] for i in event_stack])
        else: 
            raise Exception('wrong event stack')

    for entry_tmp in event_cloud:
        maxlen = entry_tmp.shape[1] if entry_tmp.shape[1] > maxlen else maxlen

    batched_event_cloud = np.zeros((batch, maxlen, 4), dtype=np.float32)

    for batch_idx in range(batch):
        lens = event_cloud[batch_idx].shape[1]
        batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

    batched_event_cloud = torch.from_numpy(batched_event_cloud.astype(np.float32))

    return batched_event_cloud


def event_conversion(event_list, time_bins, resolution, time_bins_voxel=None):
    """
    event_list: torch.tensor, B x N_max x 4, [x, y, t, p]
    """
    def sort_events(events):
        """
        events: torch.tensor, Nx4, [x, y, t, p]
        """
        e_list = sorted(events, key=lambda x: x[2])
        events = torch.stack(e_list, dim=0)

        return events

    event_list = event_list.detach().clone().cpu()

    if time_bins_voxel == None:
        time_bins_voxel = time_bins

    output = {}
    e_cnt = []
    e_voxel = []
    e_stack = []
    # e_pol_mask = []
    # e_list = []
    for entry in event_list:
        entry = sort_events(entry)
        xs = entry[:, 0]
        ys = entry[:, 1]
        ts = entry[:, 2]
        ps = entry[:, 3]
        e_cnt.append(events_to_channels(xs, ys, ps, sensor_size=resolution))
        e_voxel.append(events_to_voxel(xs, ys, ts, ps, time_bins_voxel, sensor_size=resolution))
        e_stack.append(events_to_stack_no_polarity(xs, ys, ts, ps, time_bins, sensor_size=resolution))
        # e_pol_mask.append(events_polarity_mask(ps))
        # e_list.append(torch.stack([ts, ys, xs, ps], dim=1))
    output['e_cnt'] = torch.stack(e_cnt)
    output['e_voxel'] = torch.stack(e_voxel)
    output['e_stack'] = torch.stack(e_stack)
    # output['e_pol_mask'] = torch.stack(e_pol_mask)
    # output['e_list'] = torch.stack(e_list)

    return output


def event_restore(events, resolution):
    """
    events: BxNx4, [x, y, t, p]
    """
    def convert_polarity(ps):
        """
        ps: torch.tensor, [N], -1~+1
        """
        pos_mask = ps > 0
        neg_mask = ps < 0
        ps[pos_mask] = 1
        ps[neg_mask] = -1
        return ps

    events = events.detach().cpu().clone()
    x = events[:, :, 0] * resolution[1]
    y = events[:, :, 1] * resolution[0]
    t = events[:, :, 2]
    p = events[:, :, 3]
    p = convert_polarity(p)

    return torch.stack([x, y, t, p], dim=2)


def sparse2event(sparse_tensor, time_bins, resolution, time_bins_voxel=None):
    """
    sparse_tensor: ME.SparseTensor; coords: Nx4, [b, x, y, t]; feats: Nx1, polarity
    """
    def convert_polarity(ps):
        """
        ps: torch.tensor, [N], -1~+1
        """
        pos_mask = ps > 0
        neg_mask = ps < 0
        ps[pos_mask] = 1
        ps[neg_mask] = -1
        return ps

    if time_bins_voxel == None:
        time_bins_voxel = time_bins

    batch_coords, batch_feats = sparse_tensor.decomposed_coordinates_and_features

    output = {}
    e_cnt = []
    e_voxel = []
    e_stack = []
    # e_pol_mask = []
    # e_list = []
    for coords, feats in zip(batch_coords, batch_feats):
        coords = coords.cpu()
        feats = feats.cpu()
        elist = torch.cat([coords, feats], dim=1) # Nx4, [xs, ys, ts, ps]
        elist = sorted(elist, key=lambda x: x[2]) # sorted by time
        elist = torch.stack(elist, dim=0) # Nx4, [xs, ys, ts, ps]
        xs, ys, ts, ps = elist[:, 0], elist[:, 1], elist[:, 2], elist[:, 3]
        ps = convert_polarity(ps)
        ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
        e_cnt.append(events_to_channels(xs, ys, ps, sensor_size=resolution))
        e_voxel.append(events_to_voxel(xs, ys, ts, ps, time_bins_voxel, sensor_size=resolution))
        e_stack.append(events_to_stack_no_polarity(xs, ys, ts, ps, time_bins, sensor_size=resolution))
        # e_pol_mask.append(events_polarity_mask(ps))
        # e_list.append(torch.stack([ts, ys, xs, ps], dim=1))
    output['e_cnt'] = torch.stack(e_cnt)
    output['e_voxel'] = torch.stack(e_voxel)
    output['e_stack'] = torch.stack(e_stack)
    # output['e_pol_mask'] = torch.stack(e_pol_mask)
    # output['e_list'] = torch.stack(e_list)

    return output


def stack2cnt(stack):
    """
    stack: torch.tensor, BxTBxHxW
    return: torch.tensor, Bx2xHxW, 0 for positive, 1 for negtive
    """
    stack = stack.clone().detach().round().cpu()

    pos = stack.clone()
    neg = stack.clone()
    pos[pos<0] = 0
    neg[neg>0] = 0
    neg *= -1

    pos = pos.sum(1)
    neg = neg.sum(1)

    cnt = torch.stack([pos, neg], dim=1)

    return cnt


if __name__ == '__main__':

    batch = 1
    bins = 10
    sensor_size = [4, 4]

    event_stack = torch.randint(-5, 15, [batch, bins]+sensor_size).float()
    print(event_stack)

    event_cloud = python_event_redistribute_NoPolarityStack(event_stack, mode='random')
    print(event_cloud)
    print(event_cloud.size())

    event_stack1 = events_to_voxel_torch(xs=event_cloud[0, 0, :, 0],
                                         ys=event_cloud[0, 0, :, 1],
                                         ts=event_cloud[0, 0, :, 2],
                                         ps=event_cloud[0, 0, :, 3],
                                         B=bins,
                                         sensor_size=sensor_size,
                                         temporal_bilinear=False)

    print(event_stack1)

    print((event_stack-event_stack1).sum())