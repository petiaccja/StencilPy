from stencilir import ops
import stencilir as sir


def slice_size_function(is_public = False) -> ops.FuncOp:
    loc = ops.Location("<__slice_size>", 1, 1)

    name = "__slice_size"
    function_type = sir.FunctionType(
        [sir.IndexType(), sir.IndexType(), sir.IndexType()],
        [sir.IndexType()]
    )

    func = ops.FuncOp(name, function_type, is_public, loc)

    start = func.get_region_arg(0)
    stop = func.get_region_arg(1)
    step = func.get_region_arg(2)

    c0 = func.add(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
    c1 = func.add(ops.ConstantOp(1, sir.IndexType(), loc)).get_result()

    stepm1p = func.add(ops.ArithmeticOp(step, c1, ops.ArithmeticFunction.SUB, loc)).get_result()
    stepm1n = func.add(ops.ArithmeticOp(step, c1, ops.ArithmeticFunction.ADD, loc)).get_result()
    is_step_negative = func.add(ops.ComparisonOp(step, c0, ops.ComparisonFunction.LT, loc)).get_result()
    select_stepm1: ops.IfOp = func.add(ops.IfOp(is_step_negative, 1, loc))
    select_stepm1.get_then_region().add(ops.YieldOp([stepm1n], loc))
    select_stepm1.get_else_region().add(ops.YieldOp([stepm1p], loc))
    stepm1 = select_stepm1.get_results()[0]

    stop_aligned = func.add(ops.ArithmeticOp(stop, stepm1, ops.ArithmeticFunction.ADD, loc)).get_result()
    distance = func.add(ops.ArithmeticOp(stop_aligned, start, ops.ArithmeticFunction.SUB, loc)).get_result()
    size = func.add(ops.ArithmeticOp(distance, step, ops.ArithmeticFunction.DIV, loc)).get_result()
    clamped = func.add(ops.MaxOp(size, c0, loc)).get_result()
    func.add(ops.ReturnOp([clamped], loc))

    return func


def adjust_slice_trivial_function(is_public = False) -> ops.FuncOp:
    """
    Simple method for limited cases to help optimization.
    Use only when:
    - start is a constant expression >= 0
    - stop is an arbitrary expression
    - step is a constant expression > 0
    - length is an arbitrary expression (>=0)
    """
    loc = ops.Location("<__adjust_slice_trivial>", 1, 1)

    name = "__adjust_slice_trivial"
    function_type = sir.FunctionType(
        [sir.IndexType(), sir.IndexType(), sir.IndexType(), sir.IndexType()],
        [sir.IndexType(), sir.IndexType()]
    )

    func = ops.FuncOp(name, function_type, is_public, loc)

    start = func.get_region_arg(0)
    stop = func.get_region_arg(1)
    length = func.get_region_arg(3)

    c0 = func.add(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
    is_stop_negative = func.add(ops.ComparisonOp(stop, c0, ops.ComparisonFunction.LT, loc)).get_result()
    stop_incr = func.add(ops.ArithmeticOp(stop, length, ops.ArithmeticFunction.ADD, loc)).get_result()
    stop_zero_clamped = func.add(ops.MaxOp(c0, stop_incr, loc)).get_result()
    stop_length_clamped = func.add(ops.MinOp(stop, length, loc)).get_result()
    stop_adj_op: ops.IfOp = func.add(ops.IfOp(is_stop_negative, 1, loc))
    stop_adj_op.get_then_region().add(ops.YieldOp([stop_zero_clamped], loc))
    stop_adj_op.get_else_region().add(ops.YieldOp([stop_length_clamped], loc))
    func.add(ops.ReturnOp([start, stop_adj_op.get_results()[0]], loc))

    return func


def adjust_slice_function(is_public = False) -> ops.FuncOp:
    loc = ops.Location("<__adjust_slice>", 1, 1)

    name = "__adjust_slice"
    function_type = sir.FunctionType(
        [sir.IndexType(), sir.IndexType(), sir.IndexType(), sir.IndexType()],
        [sir.IndexType(), sir.IndexType()]
    )
    func = ops.FuncOp(name, function_type, is_public, loc)

    c0 = func.add(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
    cm1 = func.add(ops.ConstantOp(-1, sir.IndexType(), loc)).get_result()

    start = func.get_region_arg(0)
    stop = func.get_region_arg(1)
    step = func.get_region_arg(2)
    length = func.get_region_arg(3)

    def lt(lhs, rhs):
        return func.add(ops.ComparisonOp(lhs, rhs, ops.ComparisonFunction.LT, loc)).get_result()

    def gte(lhs, rhs):
        return func.add(ops.ComparisonOp(lhs, rhs, ops.ComparisonFunction.GTE, loc)).get_result()

    def add(lhs, rhs):
        return func.add(ops.ArithmeticOp(lhs, rhs, ops.ArithmeticFunction.ADD, loc)).get_result()

    def select(cond, lhs, rhs):
        select_op: ops.IfOp = func.add(ops.IfOp(cond, 1, loc))
        select_op.get_then_region().add(ops.YieldOp([lhs], loc))
        select_op.get_else_region().add(ops.YieldOp([rhs], loc))
        return select_op.get_results()[0]

    start_adj = select(
        lt(start, c0),
        select(
            lt(add(start, length), c0),
            select(lt(step, c0), cm1, c0),
            add(start, length)
        ),
        select(
            gte(start, length),
            select(lt(step, c0), add(length, cm1), length),
            start
        )
    )

    stop_adj = select(
        lt(stop, c0),
        select(
            lt(add(stop, length), c0),
            select(lt(step, c0), cm1, c0),
            add(stop, length)
        ),
        select(
            gte(stop, length),
            select(lt(step, c0), add(length, cm1), length),
            stop
        )
    )

    func.add(ops.ReturnOp([start_adj, stop_adj], loc))
    return func
