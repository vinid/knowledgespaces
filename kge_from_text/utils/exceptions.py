

class ModelAlreadyExistsException(Exception):
    print("Model Already Exists. Use Override Option")

class ModelAlreadyLoadedException(Exception):
    print("ModelAlreadyLoaded. Create New Object")
