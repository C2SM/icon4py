import abc
import datetime
from typing import Protocol


class Component(Protocol):
    """
    Base class for components of the model.

    Components are the building blocks of the model. They can modify the state, return new state objects,
    store state for later usage, or perform other actions.
    """


    
    def input_properties(self) -> dict:
        """Return a dictionary of input properties for the component."""
        return {}
    
    def output_properties(self) -> dict:
        """Return a dictionary of output properties for the component."""
        return {}
    
    def __str__(self):
        return f"instance of {self.__class__}(Component)"
    
    
    def __call__(self, state, time_step:datetime.datetime):
        """
        Gets the input_properties from the model state and applies the `run` to them and
        returns the output_properties.
        
        Args:
            state: dict  model state dictionary
            time_step: datetime.datetime  current simulation time
        Returns:
            output_properties a dicationary of output quantities
        Raises:
              IncompleteStateError if the input_properties are not in the state
                
        """
        
    @abc.abstractmethod    
    def run(self, input_properties: dict, time_step:datetime.datetime) -> dict:
        """
        Runs the component on the input_properties and returns the output_properties.
        
        Args:
            input_properties: dict  input properties dictionary
            time_step: datetime.datetime  current simulation time    
        
        """
        pass