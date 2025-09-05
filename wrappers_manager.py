from pathlib import Path


def get_wrappers_list():
    """Get list of available wrapper methods by scanning wrapper files."""
    wrapper_dir = Path(__file__).parent / 'wrappers'
    wrappers = []

    for file in wrapper_dir.glob('*_wrapper.py'):
        if file.name not in ['wrapper.py', 'example_wrapper.py']:
            wrapper_name = file.stem.replace('_wrapper', '')
            wrappers.append(wrapper_name)
    return sorted(wrappers)


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
    else:
        raise ValueError('Wrappers supported: {}'.format(get_wrappers_list()))

    return wrapper
