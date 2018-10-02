from tensorflow.nn.rnn_cell import LSTMStateTuple as LSTMStateTuple


def synchronous_flatten(*nested):
    if not isinstance(nested[0], (tuple, list, dict)):
        return [[n] for n in nested]
    output = [list() for _ in nested]
    if isinstance(nested[0], dict):
        for k in nested[0].keys():
            flattened = synchronous_flatten(*[n[k] for n in nested])
            for o, f in zip(output, flattened):
                o.extend(f)
    else:
        try:
            for inner_nested in zip(*nested):
                flattened = synchronous_flatten(*inner_nested)
                for o, f in zip(output, flattened):
                    o.extend(f)
        except TypeError:
            print('(synchronous_flatten)nested:', nested)
            raise
    return output


def deep_zip(objects, depth, permeable_types=(list, tuple, dict)):
    # print("(deep_zip)objects:", objects)
    if depth != 0 and isinstance(objects[0], permeable_types):
        if isinstance(objects[0], (list, tuple)):
            zipped = list()
            for comb in zip(*objects):
                zipped.append(
                    deep_zip(comb, depth-1, permeable_types=permeable_types)
                )
            if isinstance(objects[0], LSTMStateTuple):
                zipped = LSTMStateTuple(
                    c=zipped[0],
                    h=zipped[1],
                )
            elif isinstance(objects[0], tuple):
                zipped = tuple(zipped)

            return zipped
        elif isinstance(objects[0], dict):
            zipped = dict()
            for key in objects[0].keys():
                values = [obj[key] for obj in objects]
                zipped[key] = deep_zip(values, depth-1, permeable_types=permeable_types)
            return zipped
    return tuple(objects)


def apply_func_on_depth(obj, func, depth, permeable_types=(list, tuple, dict)):
    if depth != 0 and isinstance(obj, permeable_types):
        if isinstance(obj, (list, tuple)):
            processed = list()
            for elem in obj:
                processed.append(apply_func_on_depth(elem, func, depth-1, permeable_types=permeable_types))
            if isinstance(obj, LSTMStateTuple):
                processed = LSTMStateTuple(
                    c=processed[0],
                    h=processed[1],
                )
            elif isinstance(obj, tuple):
                processed = tuple(processed)
            return processed
        elif isinstance(obj, dict):
            processed = dict()
            for key, value in obj.items():
                processed[key] = apply_func_on_depth(value, func, depth-1, permeable_types=permeable_types)
            return processed
    return func(obj)
