class EnsembleBase(object):
    """ Specifies an ensemble of ParametricSystems.  This class is intended to
    serve as a container for a list of ParametricSystems, and a convenient way
    of initialising them.  This is base class defining the API and cannot be
    used, it needs to be sub-classed, and a sub-class needs to provide a
    property called 'systems'."""
    @property
    def systems(self):
        raise NotImplementedError
