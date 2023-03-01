from stencilpy.error import *
from stencilpy import concepts


class NodeTransformer:
    def visit(self, node: Any):
        class_name = node.__class__.__name__
        handler_name = f"visit_{class_name}"
        if hasattr(self.__class__, handler_name):
            return getattr(self.__class__, handler_name)(self, node)
        self.on_invalid_node(node)

    def on_invalid_node(self, node: Any):
        loc = concepts.Location.unknown()
        if hasattr(node, "location") and isinstance(node.location, concepts.Location):
            loc = node.location
        class_name = node.__class__.__name__
        raise InternalCompilerError(loc, f"no visitor implemented for node of type `{class_name}`")
