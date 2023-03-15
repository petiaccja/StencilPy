from stencilir import ops
import stencilir as sir
from stencilpy.compiler.symbol_table import SymbolTable
from stencilpy.compiler.node_transformer import NodeTransformer
from stencilpy.compiler import hlast
import ctypes
from .utility import as_sir_loc


class SirOpTransformer(NodeTransformer):
    region_stack: list[ops.Region] = None
    current_region: ops.Region = None
    symtable: SymbolTable = None

    def push_region(self, region: ops.Region):
        self.region_stack.append(region)
        self.current_region = region

    def pop_region(self):
        self.region_stack.pop(len(self.region_stack) - 1)
        self.current_region = self.region_stack[-1] if self.region_stack else None

    def __init__(self):
        self.symtable = SymbolTable()
        self.region_stack = []

    def visit_slice(self, slc: hlast.Slice) -> tuple[ops.Value, ops.Value, ops.Value]:
        if slc.single:
            loc = as_sir_loc(slc.lower.location)

            def as_index(expr: ops.Value) -> ops.Value:
                return self.current_region.add(ops.CastOp(expr, sir.IndexType(), loc)).get_result()

            c1 = self.current_region.add(ops.ConstantOp(1, sir.IndexType(), loc)).get_result()
            start = as_index(self.visit(slc.lower)[0])
            stop = self.current_region.add(ops.ArithmeticOp(start, c1, ops.ArithmeticFunction.ADD, loc)).get_result()
            step = c1
            return start, stop, step
        else:
            loc = ops.Location("<slice_autofill>", 1, 1)
            constant_pos_step = False
            index_min = -2 ** (ctypes.sizeof(ctypes.c_void_p) * 8 - 1)
            index_max = -(index_min + 1)
            if slc.step:
                constant_pos_step = isinstance(slc.step, hlast.Constant) and slc.step.value > 0
                step = self.visit(slc.step)[0]
            else:
                constant_pos_step = True
                step = self.current_region.add(ops.ConstantOp(1, sir.IndexType(), loc)).get_result()

            if slc.lower:
                start = self.visit(slc.lower)[0]
            else:
                start = (
                    self.current_region.add(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
                    if constant_pos_step
                    else self.current_region.add(ops.ConstantOp(index_min, sir.IndexType(), loc)).get_result()
                )

            if slc.upper:
                stop = self.visit(slc.upper)[0]
            else:
                stop = (
                    self.current_region.add(ops.ConstantOp(index_max, sir.IndexType(), loc)).get_result()
                    if constant_pos_step
                    else self.current_region.add(ops.ConstantOp(index_min, sir.IndexType(), loc)).get_result()
                )

            return start, stop, step
