import abc
import datetime
from typing import Protocol

from icon4py.model.common import exceptions
from icon4py.model.common.states import model


class Component(Protocol):
    """
    Protocol for components of the model.

    Components are the building blocks of the model. 
    The operate on model state variables and transform it or produce new state variables.

    Each component should declare its inputs and outputs in terms of name, expected units and dimensions.

    Components are Callables upon __call__ the get passed a reference for the model state and
    select their needed inputs from there.

    In order to define a component, subclass from this one and implement the `run` method.
    
    TODO (@halungge): add more consistency checks.
     - check for mathching units and allow for unit conversion
     - check for consistency of dimensions of state and input_properties
            
    """

  
    @abc.abstractmethod
    @property
    def input_properties(self) -> dict[str, model.FieldMetaData]:
        """Return a dictionary of input properties for the component:contains name, units and dimension of 
        the output_properties"""
        raise NotImplementedError
    
    @abc.abstractmethod
    @property
    def output_properties(self) -> dict[str, model.FieldMetaData]:
        """Returns a dictionary of the output of the component: contains name, units and dimension of 
        the output_properties.
        
        TODO (@halungge): is this too generic and we should split into separate properties for 
            tendencies, diagnostics, prognostics, etc? Along the same lines how should we track the time
            the produced values are valid for?
        """
        raise NotImplementedError
    
    def __str__(self):
        return (f"instance of {self.__class__}(Component) uses inputs: "
                f"{self.input_properties.keys()} \n "
                f"produces : {self.output_properties.keys()}")

    
    
    def __call__(self, state:dict[str, model.DataField], time_step:datetime.datetime):
        """
        Gets the input_properties from the model state and applies the `run` to them and
        returns the output_properties.
        This function should implement some general functionality like checks
        
        Args:
            state: dict  model state dictionary
            time_step: datetime.datetime  current simulation time
        Returns:
            output_properties a dictionary of output quantities
        Raises:
              IncompleteStateError if the input_properties are not in the state
                
        """
        for key in self.input_properties.keys():
            if key not in state:
                raise exceptions.IncompleteStateError(f"input_properties {key} not found in state")
            #TODO (@halungge) (check that units, dimensions match state[key] and input_properties[key]) if they do not match 
        
        return self.run(state, time_step)
        
    @abc.abstractmethod    
    def run(self, state: dict[str, model.DataField], time_step:datetime.datetime) -> dict[str, model.DataField]:
        """
        Runs the component on the input_properties and returns the output_properties.

        This function *must* to be implemented in each implementing class to contain the 
        real logic of the component.

        TODO (@halungge): is it possible to improve this interface not haveing to pass on the entire state for example?

        Args:
            state: dict  input properties dictionary
            time_step: datetime.datetime  current simulation time
        Returns: 
            dict  output properties dictionary
        """
        raise NotImplementedError