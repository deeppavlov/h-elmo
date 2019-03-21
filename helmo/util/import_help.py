from helmo.util import interpreter
interpreter.extend_python_path_for_project()


def form_load_cmd(file_name, obj_name, imported_as):
    file_name.replace('/', '.')
    return "from helmo.nets.%s import %s as %s" % (file_name, obj_name, imported_as)
