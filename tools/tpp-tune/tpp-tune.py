#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Union
from pprint import pprint


# Enable automagically finding TPP-MLIR's python modules (which include
# and extend MLIR's Python bindings).
python_packages_path = Path(__file__).parent.parent / "python_packages"
if python_packages_path.exists():
    sys.path = [str(python_packages_path)] + sys.path


from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import tune as transform_tune


if len(sys.argv) == 1:
    file = sys.stdin
else:
    filename = sys.argv[1]
    if filename == "-":
        file = sys.stdin
    else:
        file = open(sys.argv[1])


def walker(f):
    def inner(op: Union[ir.OpView, ir.Operation]):
        if f(op):
            pass
        else:
            for region in op.regions:
                for block in region.blocks:
                    for child_op in block:
                        inner(child_op)

    return inner


with ir.Context(), ir.Location.unknown():
    schedule = ir.Module.parse(file.read())

    choices = {}

    @walker
    def choices_finder(op):
        if isinstance(op, transform_tune.TuneSelectOp):
            choices[op.name] = tuple(op.options)
            return True

    choices_finder(schedule.operation)

    pprint(choices, stream=sys.stderr)

    # Aint tuning easy!!
    selected = {key: values[0] for key, values in choices.items()}

    @walker
    def selected_rewriter(op: Union[ir.OpView, ir.Operation]):
        if isinstance(op, transform_tune.TuneSelectOp):
            with ir.InsertionPoint(op):
                param = transform.param_constant(
                    transform.AnyParamType.get(), selected[op.name]
                )
                for use in op.result.uses:
                    use.owner.operands[use.operand_number] = param
            return True

    selected_rewriter(schedule.operation)

    print(schedule)
