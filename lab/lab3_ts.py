import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ir import *
from codegen.cpu import *

import codegen.cpu
from core.ir import *
from core.ast import *

def get_obj(ir: (Index, Scalar)):
    obj = ir
    while hasattr(obj, 'dobject'):
        obj = obj.dobject
    return obj

def replace_index_with_scalar(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            replace_index_with_scalar(l, old, new)
    elif type(ir) == Loop:
        replace_index_with_scalar(ir.body, old, new)
    elif type(ir) == Expr:
        if type(ir.left) in (Index, Scalar):
            obj = get_obj(ir.left)
            if obj == old:
                ir.left = new
        else:
            replace_index_with_scalar(ir.left, old, new)
        
        if type(ir.right) in (Index, Scalar):
            obj = get_obj(ir.right)
            if obj == old:
                ir.right = new
        else:
            replace_index_with_scalar(ir.right, old, new)
    elif type(ir) == Assignment:
        if type(ir.lhs) in (Index, Scalar):
            obj = get_obj(ir.lhs)
            if obj == old:
                ir.lhs = new
        else:
            replace_index_with_scalar(ir.lhs, old, new)
        
        if type(ir.rhs) in (Index, Scalar):
            obj = get_obj(ir.rhs)
            if obj == old:
                ir.rhs = new
        else:
            replace_index_with_scalar(ir.rhs, old, new)

def rebind_iterate(ir, old_idx, new_idx):
    if isinstance(ir, (list, tuple)):
        for l in ir:
            rebind_iterate(l, old_idx, new_idx)
    elif isinstance(ir, Assignment):
        rebind_iterate(ir.lhs, old_idx, new_idx)
        rebind_iterate(ir.rhs, old_idx, new_idx)
    elif isinstance(ir, Index):
        if ir.index == old_idx:
            ir.index = new_idx
    elif isinstance(ir, Loop):
        rebind_iterate(ir.body, old_idx, new_idx)

def fusable_level(node0, node1):
    def _fusable_level(loop0, loop1, level):
        if  type(loop0)!=Loop or type(loop1)!=Loop:
            return level
        if loop0.start==loop1.start and loop0.end==loop1.end and loop0.step==loop1.step:
            return _fusable_level(loop0.body[0], loop1.body[0], level+1)
        else:
            return level
    loop0 = node0.compute[0]
    loop1 = node1.compute[0]
    return _fusable_level(loop0, loop1, 0)

def move_ir(node0, node1, move_level):
    def _move_ir(loop0, loop1, cur_level):
        if cur_level==move_level-1:
            loop1.body = loop0.body + loop1.body 
        else:
            _move_ir(loop0.body[0], loop1.body[0], cur_level+1) 

    loop0 = node0.compute[0]
    loop1 = node1.compute[0]
    _move_ir(loop0, loop1, 0)

def get_rhs(node):
    def _get_rhs(ir):
        if type(ir)==Loop:
            return _get_rhs(ir.body[0])
        else:
            print(ir)
            assert type(ir)==Assignment
            return ir.rhs
    return _get_rhs(node.compute[0])

def fuse(ast_wit_ir):
    elementwise_op = op_mapping
    node = ast_wit_ir
    
    def find_fusable_pairs(node, res):
        if type(node) == TensorOp and node.op_type in elementwise_op:
            if len(node.operators) >= 2:
                print(f"Type of operators[0]: {type(node.operators[0])}, Length: {len(node.operators)}")
                if type(node.operators[0]) == TensorOp and node.operators[0].op_type in elementwise_op:      
                    print("Find fusable pairs! Left")
                    fuse_tensor_ops(node, node.operators[0])
                    
                print(f"Type of operators[1]: {type(node.operators[1])}, Length: {len(node.operators)}")
                if type(node.operators[1]) == TensorOp and node.operators[1].op_type in elementwise_op:
                    print("Find fusable pairs! Right")
                    fuse_tensor_ops(node, node.operators[1])

    def fuse_tensor_ops(parent_node, child_node):
        level = fusable_level(parent_node, child_node)
        if level > 0:
            print(f"Fusing at level {level}")

            # Make sure child_node.compute has elements before accessing it
            if child_node.compute and len(child_node.compute) >= 2:
                rhs = get_rhs(child_node)

                # Make sure parent_node.compute has elements before accessing it
                if parent_node.compute and len(parent_node.compute) >= 2:
                    # Replace indices in the child_node's RHS with the corresponding indices in the parent_node
                    replace_index_with_scalar(rhs, child_node.compute[1], parent_node.compute[1])

                # Move the body of the child_node into the body of the parent_node at the fusable level
                move_ir(child_node, parent_node, level)

                # Update the size information of the parent_node
                parent_node.fix_size.extend(child_node.fix_size)
                parent_node.ref_size.extend(child_node.ref_size)

                # Make sure parent_node.compute has elements before accessing it
                if parent_node.compute:
                    # Update the compute information of the parent_node
                    parent_node.compute = parent_node.compute + child_node.compute[1:]

                # Additional logic: You can perform further optimization or cleanup here
                # For example, removing the child_node from the parent_node's decl list

                # Remove the child_node from the parent_node's decl list
                parent_node.decl.remove(child_node)

                # Ensure that the parent_node's dtype is updated if needed
                parent_node.dtype = child_node.dtype

                print(f"Parent Node Compute Length: {len(parent_node.compute)}")
                print(f"Child Node Compute Length: {len(child_node.compute)}")
                print(f"Parent Node: {parent_node}")
                print(f"Child Node: {child_node}")

    t = helpers.Traversal(find_fusable_pairs)
    t(node)
    return node

def test1():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))
    E = Tensor('e', (30, 30))

    res1 = A + B 
    res2 = C + D
    res = res1 + res2 + E
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print(code)
    new_res_with_ir = fuse(res_with_ir)
    print(new_res_with_ir)

def test2():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))

    res = A + B + C
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print(code)
    new_res_with_ir = fuse(res_with_ir)
    print(new_res_with_ir)

if __name__ == "__main__":
    test1()