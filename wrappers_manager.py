def get_wrappers_list():
    return ['disk', 'superpoint', 'ripe', 'dedode', 'dedode-G', 'aliked',  'random', 'no_wrapper']


def wrappers_manager(name, device, mismatches=0):
    print(f'Creating wrapper for {name} on device {device}.\n')

    if name == 'disk':
        from wrappers.disk_wrapper import DiskWrapper
        wrapper = DiskWrapper(device=device)
        wrapper.name = name
    
    elif name == 'superpoint':
        from wrappers.superpoint_wrapper import SuperPointWrapper
        wrapper = SuperPointWrapper(device=device)
        wrapper.name = name

    elif name == 'ripe':
        from wrappers.ripe_wrapper import RIPEWrapper
        wrapper = RIPEWrapper(device=device)
        wrapper.name = name
    
    elif name == 'dedode' or name == 'dedode-B':
        from wrappers.dedode_wrapper import DeDoDeWrapper
        wrapper = DeDoDeWrapper(device=device)
        wrapper.name = name
    
    elif name == 'dedode-G':
        from wrappers.dedode_wrapper import DeDoDeWrapper
        wrapper = DeDoDeWrapper(device=device, descriptor_G=True)
        wrapper.name = name
    
    elif name == 'aliked':
        from wrappers.aliked_wrapper import AlikedWrapper
        wrapper = AlikedWrapper(device=device)
        wrapper.name = name
    
    elif name == 'sift':
        from wrappers.sift_wrapper import SIFTPyColmapWrapper
        wrapper = SIFTPyColmapWrapper(device=device)
        wrapper.name = name

    elif name == 'random':
        from random_wrapper import RandomPointsWrapper
        wrapper = RandomPointsWrapper(mismatch_perc=mismatches)
        wrapper.name = name
    
    # if name is a tuple
    elif '_' in name:
        from mix_kpts_desc_wrapper import MixKptsDescsWrapper
        detector, descriptor = name.split('_')
        wrapper = MixKptsDescsWrapper(detector=detector, descriptor=descriptor, device=device)
        wrapper.name = name
    
    elif name == 'no_wrapper':
        from no_wrapper import NoWrapper
        wrapper = NoWrapper(device=device)
        wrapper.name = name

    else:
        raise ValueError('Wrappers supported: {}'.format(get_wrappers_list()))
    
    return wrapper