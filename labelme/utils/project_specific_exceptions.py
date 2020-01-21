
class EdgeLabelingError(Exception):
    '''
    Base class for all edge labeling errors
    '''

class EdgeNotFound(EdgeLabelingError):
    '''
    Raised when an edge is not found by an edge adjustment routine
    '''
    pass

class PointsIncorrectOrder(EdgeLabelingError):
    '''
    Points are expected in a specific order, when they are not in this order, raise an error
    '''
    pass