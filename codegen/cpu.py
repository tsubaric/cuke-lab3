from pathlib import Path
from core.ast2ir import *
from core import helpers
from ext.set import *


def to_string(ir):
    match ir.__class__.__name__:
        case 'Expr':
            return to_string(ir.left) + f" {ir.op} " + to_string(ir.right)
        case 'Assignment':
            if ir.op is None:
                return f"{to_string(ir.lhs)} = {to_string(ir.rhs)};\n"
            else:
                return f"{to_string(ir.lhs)} {ir.op}= {to_string(ir.rhs)};\n"
        case 'Loop':
            code = f"for (int {to_string(ir.iterate)} = {to_string(ir.start)}; {to_string(ir.iterate)} < {to_string(ir.end)}; {to_string(ir.iterate)} += {to_string(ir.step)}) {{\n"
            for e in ir.body:
                if e:
                    code += to_string(e)
            code += "} \n"
            return code
        case 'Scalar' | 'Ndarray' | 'Ref':
            return ir.name()
        case 'Index':
            assert ir.index != None
            if ir.ind_arr != None:
                if type(ir.ind_arr) == Slice:
                    return f'{to_string(ir.dobject)}[(({to_string(ir.ind_arr.start)})+({to_string(ir.ind_arr.step)})*({to_string(ir.index)}))]'
                else:
                    return f'{to_string(ir.dobject)}[{to_string(ir.ind_arr[to_string(ir.index)])}]'
            else:
                return f'{to_string(ir.dobject)}[{to_string(ir.index)}]'
        case 'Min':
            return f"Min({to_string(ir.first)}, {to_string(ir.second)})"
        case 'Max':
            return f"Max({to_string(ir.first)}, {to_string(ir.second)})"
        case 'Round':
            return f"Round({to_string(ir.expr)})"
        case 'Decl':
            # variables are passed in as pytorch arguments
            if type(ir.dobject) == Scalar:
                if not ir.dobject.is_arg:
                    # it is a zero or one
                    if ir.dobject.val != None:
                        return f"{ir.dobject.dtype} {ir.dobject.name()} = {to_string(ir.dobject.val)};\n"
                    else:
                        return f"{ir.dobject.dtype} {ir.dobject.name()};\n"
                else:
                    return ''
            elif type(ir.dobject) == Ndarray:
                code = ''
                if not ir.dobject.is_arg:
                    if ir.dobject.val != None:
                        code = f'torch::Tensor obj_{ir.dobject.name()} = torch::{"ones" if ir.dobject.val == 1 else "zeros"}({{{",".join([to_string(s) for s in ir.dobject.size])}}}, at::k{"Int" if ir.dobject.dtype=="int" else "Float"});\n'
                    else:
                        code = f'torch::Tensor obj_{ir.dobject.name()} = torch::empty({{{",".join([to_string(s) for s in ir.dobject.size])}}}, at::k{"Int" if ir.dobject.dtype=="int" else "Float"});\n'

                code += f'auto {ir.dobject.name()} = obj_{ir.dobject.name()}.accessor<{ir.dobject.dtype}, {len(ir.dobject.size)}>();\n'
                return code
            elif type(ir.dobject) == Ref:
                code = f'{ir.dobject.dobject.dtype}* {ir.dobject.name()} = ({ir.dobject.dobject.dtype}*)&{ir.dobject.dobject.addr()}'
                return code
        case _:
            return str(ir)


def gen_cpp(ast):
    code = ''
    def action(node, res):
        if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor or type(node) == Set:
            res.extend(node.decl)
        elif type(node) == TensorOp:
            res.extend(node.decl)
            res.extend(node.compute)

    t = helpers.Traversal(action)
    ir = t(ast)

    args = helpers.get_input_nodes(ast)
    args = ', '.join([f'torch::Tensor obj_{a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])

    for d in ir:
        if d:
            code += to_string(d)


    if type(ast.eval) == Scalar:
        rtype = ast.dtype
        code += f'return {ast.eval.name()};\n'
    elif type(ast.eval) == Ndarray:
        rtype = 'torch::Tensor'
        code += f'return obj_{ast.eval.name()};\n'
    else:
        raise TypeError('wrong output type')
    
    file_path = Path('./codegen/cpp_template.cpp')
    with open(file_path, 'r') as f:
        c_code = f.read()
        c_code = c_code.replace('RTYPE', rtype).replace('FNAME', ast.name).replace('ARGS', args).replace('CODE', code)
    return c_code

    # with open('codegen/cpp_template.cpp', 'r') as f:
    #     c_code = f.read()
    #     c_code = c_code.replace('RTYPE', rtype).replace('FNAME', ast.name).replace('ARGS', args).replace('CODE', code)
    # return c_code
