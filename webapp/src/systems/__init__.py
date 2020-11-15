from systems.rnn_system import RnnSystem

def get_system(name, args, schema=None, timed=False, model_path=None, markable_detector_path=None):
    if name == 'yourmodel':
        raise ValueError('System not defined yet')
    elif name == 'full':
        return RnnSystem(name, args, model_path, markable_detector_path, timed)
    elif name == 'uabaseline':
        return RnnSystem(name, args, model_path, markable_detector_path, timed)
    else:
        raise ValueError('Unknown system %s' % name)
