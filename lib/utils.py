import importlib.util, sys

def from_path_import(name=None, path=None, globals=None, demands=[]):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    
    for demand in demands:
        globals[demand] = getattr(module, demand)