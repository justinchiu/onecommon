from systems.rnn_system import RnnSystem, PomdpSystem

def get_system(name, args, schema=None, timed=False, model_path=None, markable_detector_path=None, inference_args=None):
    if name == 'yourmodel':
        raise ValueError('System not defined yet')
    elif name == 'pomdp':
        return PomdpSystem(name, args, timed, inference_args)
    elif name == 'full':
        return RnnSystem(name, args, model_path, markable_detector_path, timed, inference_args)
    elif name == 'uabaseline':
        return RnnSystem(name, args, model_path, markable_detector_path, timed, inference_args)
    else:
        raise ValueError('Unknown system %s' % name)
