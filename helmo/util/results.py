import os

from helmo.util import interpreter
interpreter.extend_python_path_for_project()


def load_tt_results(result_dir, required_metric_names):
    folders = filter(lambda x: x.isdigit(), os.listdir(result_dir)) if os.path.exists(result_dir) else []
    metrics = list()
    launches_for_testing = list()
    trained_launches = list()
    for folder in folders:
        one_launch_metrics = dict()
        path_to_metrics = os.path.join(result_dir, folder, 'testing/results')
        if os.path.exists(path_to_metrics):
            for metric_file in os.listdir(path_to_metrics):
                metric_name = metric_file.split('_')[0]
                with open(os.path.join(path_to_metrics, metric_file), 'r') as f:
                    one_launch_metrics[metric_name] = float(f.read())
        if set(required_metric_names) == set(one_launch_metrics.keys()):
            metrics.append(one_launch_metrics)
            trained_launches.append(folder)
        elif len(one_launch_metrics) > 0:
            launches_for_testing.append(folder)
            trained_launches.append(folder)
    return metrics, launches_for_testing, trained_launches
