from ABC import ABC

class Task(ABC):
    
    def __init__(self, name, params):
        self.name = name
        self.params = params
        
    def run(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    