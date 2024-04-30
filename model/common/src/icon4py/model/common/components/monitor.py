import abc
from abc import ABC
from datetime import datetime


class Monitor(ABC):
    """
    Monitor component of the model.

    Monitor is a base class for components that store or freeze state for later usage but don't modify it or return any new state objects.

    Named after Sympl Monitor component: https://sympl.readthedocs.io/en/latest/monitors.html
    """

    def __str__(self):
        return "instance of {}(Monitor)".format(self.__class__)

    @abc.abstractmethod
    def store(self, state: dict, model_time: datetime, *args, **kwargs):
        """Store state and perform class specific actions on it.


        Args:
            state: dict  model state dictionary
        """
        pass
