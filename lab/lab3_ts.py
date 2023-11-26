import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ir import *
from codegen.cpu import *

import codegen.cpu
from core.ir import *
from core.ast import *

#====================================================================> PrintCCode
def PrintCCode(ir):
    code = ''
    for d in ir:
        # if d:
        code += to_string(d)
    print(code)

#==========================================================================================================> My Helper functions
#====================================================================> FindBody
def FindBody(nested_loop):
    if not type(nested_loop) == Loop:
        return nested_loop
    if type(nested_loop.body[0]) == Loop:
        return FindBody(nested_loop.body[0])
    else:
        return nested_loop.body

#====================================================================> get_operation
#input = TensorOp -> operators[].compute
#output = "+", "@"
def get_operation(tensor_op):
    loop_plus_body = tensor_op[0]
    body = FindBody(loop_plus_body)
    ret = to_string((body[0].rhs).op)
    return ret

#====================================================================> count_loops
#input = TensorOp -> operators[].compute[0]
#output = number of loops
def count_loops(tensor_op_0, counter):
    if not type(tensor_op_0) == Loop:
        return counter
    if type(tensor_op_0.body[0]) == Loop:
        counter = counter + 1
        return count_loops(tensor_op_0.body[0], counter)
    else:
        counter = counter + 1
        return counter
    
#====================================================================> Loop lower/upper bounds
#input = TensorOp -> operators[].compute[0]
#output = all loops upper bounds
def get_bounds(tensor_op_0, lower_list, upper_list, steps):
    if not type(tensor_op_0) == Loop:
        return (lower_list, upper_list, steps)
    if type(tensor_op_0.body[0]) == Loop:
        lower_list.append(tensor_op_0.start)
        upper_list.append(tensor_op_0.end)
        steps.append(tensor_op_0.step)
        return get_bounds(tensor_op_0.body[0], lower_list, upper_list, steps)
    else:
        lower_list.append(tensor_op_0.start)
        upper_list.append(tensor_op_0.end)
        steps.append(tensor_op_0.step)
        return (lower_list, upper_list, steps)

#====================================================================> get_loop_iterate i.e _l6, _l7
def get_loop_iterate(loops, iterate_list):
    if not type(loops) == Loop:
        return iterate_list
    if type(loops.body[0]) == Loop:
        iterate_list.append(loops.iterate)
        return get_loop_iterate(loops.body[0], iterate_list)
    else:
        iterate_list.append(loops.iterate)
        return iterate_list

#====================================================================> get_body_Indices_helper
def get_body_Indices_helper(line_index_list):
    # PrintCCode(line_index_list)
    for i in range(0,len(line_index_list)):
        if type(line_index_list[i]) == Expr:
            return True
    return False

#====================================================================> get_body_Indices [[arr14[_l6][_l7], arr10[_l6][_l7],...], [] ] ???????????? hard code
def get_body_Indices(loop_body):
    ind_list = []
    for i in range(0,len(loop_body)):
        line_index = []
        line = loop_body[i]
        
        if type(line.lhs) == Index:
            line_index.append(line.lhs)

        if type(line.rhs) == Expr:
            line_index.append(line.rhs.left)
            line_index.append(line.rhs.right)
        
        while(get_body_Indices_helper(line_index)):
            temp_list = []
            for i in range(0,len(line_index)):
                if type(line_index[i]) == Expr:
                    temp_list.append(line_index[i].left)
                    temp_list.append(line_index[i].right)
                elif type(line_index[i]) == Index:
                    temp_list.append(line_index[i])
            
            line_index = temp_list
        if len(line_index) != 0:
            ind_list.append(line_index)

    return ind_list

#====================================================================> get_index_helper
def get_index_helper(one_Index_obj, ind_list):
    if type(one_Index_obj) == Ndarray:
        return ind_list
    
    ind_list.append(one_Index_obj.index)
    get_index_helper(one_Index_obj.dobject, ind_list)
    return ind_list

#====================================================================> get index: [[arr14[_l6][_l7],arr10[_l6][_l7],e[_l6][_l7]], [], ... ] -> [[[_l7,_l6], [_l7,_l6], [_l7,_l6]], [], ....]
def get_index_of_all_indices(index_list):
    result_list = []
    for i in range(0, len(index_list)):
        one_line_indices = index_list[i]
        temp_list = []
        for j in range(0, len(one_line_indices)):
            index_of_one = get_index_helper(one_line_indices[j], [])
            temp_list.append(index_of_one)

        result_list.append(temp_list)
    
    return result_list

#====================================================================> get_name_helper
def get_name_helper(one_Index_obj):
    if type(one_Index_obj) == Ndarray:
        return one_Index_obj
    
    ret = get_name_helper(one_Index_obj.dobject)
    return ret

#====================================================================> get name: [[arr14[_l6][_l7],arr10[_l6][_l7],e[_l6][_l7]], [], ... ] -> [[arr, arr7 arr8], [], ....]
def get_name_of_all_indices(index_list):
    result_list = []
    for i in range(0, len(index_list)):
        one_line_indices = index_list[i]
        temp_list = []
        for j in range(0, len(one_line_indices)):
            name_of_one = get_name_helper(one_line_indices[j])
            temp_list.append(name_of_one)

        result_list.append(temp_list)
    
    return result_list

#====================================================================> i.e [[[_l7,_l6], [_l7,_l6], [_l7,_l6]], [], ....] -> [[[2, 0], [1, 0], [2, 1]]]
def get_index_loop_level(loop_iterate_list, indices_index):
    result_list = []
    for i in range(0, len(indices_index)):
        line_list = []
        line = indices_index[i]
        for j in range(0,len(line)):
            obj_list = []
            one_obj = line[j]
            for k in range(0, len(one_obj)):
                for l in range(0,len(loop_iterate_list)):
                    if loop_iterate_list[l] == one_obj[k]:
                        obj_list.append(len(loop_iterate_list)-l-1)
            line_list.append(obj_list)
        result_list.append(line_list)
    return result_list

#===================================================================>> not all loop equal
#====================================================================> new_body
def new_body_b(nested_loop, body_to_add):
    if not type(nested_loop) == Loop:
        nested_loop = body_to_add + nested_loop
        return nested_loop
    if type(nested_loop.body[0]) == Loop:
        return new_body_b(nested_loop.body[0], body_to_add)
    else:
        ret = body_to_add + nested_loop.body
        return ret

#====================================================================> update_body ???????????? hard code
def update_body_b(nested_loop, new_body):
    loop_count = count_loops(nested_loop.compute[0], 0)

    if loop_count == 0:
        return
    if loop_count == 1:
        nested_loop.compute[0].body = new_body
    if loop_count == 2:
        nested_loop.compute[0].body[0].body = new_body
    if loop_count == 3:
        nested_loop.compute[0].body[0].body[0].body = new_body
    if loop_count == 4:
        nested_loop.compute[0].body[0].body[0].body[0].body = new_body
    if loop_count == 5:
        nested_loop.compute[0].body[0].body[0].body[0].body[0].body = new_body

    return nested_loop
#====================================================================> Move IR (from 1 -> 2) 1 will be child (for complete code transfer) edit this
def move_ir_2(tensor_op1, tensor_op2):
    body_of_1 = FindBody(tensor_op1.compute[0])
    tensor_op1.compute = []
    new_body_of_2 = body_of_1
    new_body_of_2 = new_body_b(tensor_op2.compute[0], body_of_1)
    
    tensor_op2 = update_body_b(tensor_op2, new_body_of_2)
    return (tensor_op1, tensor_op2)

#===============================================================================>> all loop equal
#====================================================================> get_ops_order
def get_ops_order(body_of_parent, replacement, kkk):
    line_p = to_string(body_of_parent[0])
    operators_p = re.findall(r'[\+\-\*/]', line_p)
    
    line_c = to_string(replacement)
    operators_c = re.findall(r'[\+\-\*/]', line_c)

    ops_seq = []
    pushed = 0
    # print("length of loop: ",len(operators_p)+len(operators_c))
    # print("Child op:", operators_c)
    # print("Parent op:", operators_p)
    cond = len(operators_p)+len(operators_c)

    iter = 0
    while iter < cond:
        if pushed == kkk - 1:
            while(len(operators_c) != 0):
                # print("innnn 1", iter)
                ops_seq.append(operators_c.pop(0))
                pushed = pushed+1
                iter = iter + 1
        else:
            ops_seq.append(operators_p.pop(0))
            pushed = pushed+1
            iter = iter + 1
    # print("---------> ops:",ops_seq)
    return ops_seq
#====================================================================> new_body_a 
def new_body_a(body_parts_of_parent, body_of_parent,replacement, parent_op, child_op, replace_this):
    left_side = body_parts_of_parent[0]
    kkk = -1
    for i in range(0, len(body_parts_of_parent)):
        if to_string(body_parts_of_parent[i]) == to_string(replace_this):
            body_parts_of_parent[i] = replacement
            kkk = i
    
    # print("kkk = ", kkk)
    # print(to_string(replacement))
    temp = to_string(replacement)
    temp_ops = re.findall(r'[\+\-\*/]', temp)
    # print(temp_ops)
    pop_time = len(temp_ops)
    
    ops_seq = get_ops_order(body_of_parent, replacement, kkk)
    
    for jj in range(0, pop_time):
        ops_seq.pop(kkk-1)

    right_side = body_parts_of_parent[1]
    for j in range(2, len(body_parts_of_parent)):
        oper = ops_seq.pop(0)
        right_side = Expr(right_side, body_parts_of_parent[j], oper)

    new_ass = Assignment(left_side,right_side)
    return new_ass

#====================================================================> update_body 
def update_body_a(nested_loop, new_body):
    loop_count = count_loops(nested_loop.compute[0], 0)
    if loop_count == 0:
        return
    if loop_count == 1:
        nested_loop.compute[0].body = [new_body]
    if loop_count == 2:
        nested_loop.compute[0].body[0].body = [new_body]
    if loop_count == 3:
        nested_loop.compute[0].body[0].body[0].body = [new_body]
    if loop_count == 4:
        nested_loop.compute[0].body[0].body[0].body[0].body = [new_body]
    if loop_count == 5:
        nested_loop.compute[0].body[0].body[0].body[0].body[0].body = [new_body]
    return nested_loop

#====================================================================> a[_l0][_l1]b[_l0][_l1] -> a[_l4][_l5]b[_l4][_l5]
def updated_assignment(index_list,loop_iterate_of_parent,levels_info):
    ret_index_list = []
    for i  in range(0, len(index_list)):
        ind = index_list[i]

        def recursive_index_update(ind, loop_iterate_of_parent, levels_info):
            # print(levels_info)
            if len(levels_info) == 0:
                return ind
            
            ind.index = loop_iterate_of_parent[levels_info.pop()]
            ind.dobject = recursive_index_update(ind.dobject, loop_iterate_of_parent, levels_info)
            return ind
        
        ind = recursive_index_update(ind, loop_iterate_of_parent, levels_info[i])
        ret_index_list.append(ind)
    return ret_index_list

#====================================================================> Move IR (from 1 -> 2) 1 will be child (for complete code transfer) edit this
def move_ir_1(tensor_op1, tensor_op2, position, parent_op, child_op):
    body_of_child = FindBody(tensor_op1.compute[0])
    body_of_parent = FindBody(tensor_op2.compute[0])
    
    temp = to_string(body_of_child[0])
    temp_ops = re.findall(r'[\+\-\*/]', temp)

    loop_iterate_of_parent = get_loop_iterate(tensor_op2.compute[0], [])
    loop_iterate_of_child = get_loop_iterate(tensor_op1.compute[0], [])
    index_list_p = get_body_Indices(body_of_parent)
    index_list = get_body_Indices(body_of_child)
    indices_index = get_index_of_all_indices(index_list)
    levels_info = get_index_loop_level(loop_iterate_of_child, indices_index)
    updated_index = updated_assignment(index_list[0],loop_iterate_of_parent, levels_info[0])
    replace_this = updated_index[0]
    replacement = updated_index[1]
    
    for i in range(2, len(updated_index)):
        replacement = replacement = Expr(replacement, updated_index[i], temp_ops.pop(0))

    tensor_op1.compute = []
    new_body_of_2 = new_body_a(index_list_p[0], body_of_parent,replacement, parent_op, child_op, replace_this)
    
    tensor_op2 = update_body_a(tensor_op2, new_body_of_2)
    
    return (tensor_op1, tensor_op2)

#====================================================================> 
def post_order_traversal(root):
    if type(root) == TensorOp:
        post_order_traversal(root.operators[0])
        post_order_traversal(root.operators[1])

        # =========================================> Right Child Check
        if (type(root.operators[0]) == TensorOp) and (root.op_type in op_mapping) and (root.operators[0].op_type in op_mapping):
            parent_op = get_operation(root.compute)
            child_op = get_operation(root.operators[0].compute)
            
            if child_op == '+' or child_op == '-' or child_op == '@' or child_op == '/':
                parent_bounds = get_bounds(root.compute[0], [], [], [])
                child_bounds = get_bounds(root.operators[0].compute[0], [], [], [])
                #==================================> For equal bounds
                if parent_bounds == child_bounds:
                    print("==============================================> Safe Left [equal bounds] + , -, /")

                    # ===> Parent
                    body_parent = FindBody(root.compute[0])
                    index_list_parent = get_body_Indices(body_parent)
                    names_parent = get_name_of_all_indices(index_list_parent)
                    name_list_p = names_parent[0]
                    # PrintCCode(name_list_p)

                    # ===> Child
                    body_child = FindBody(root.operators[0].compute[0])
                    index_list_child = get_body_Indices(body_child)
                    names_child = get_name_of_all_indices(index_list_child)
                    name_list_c = names_child[0]
                    # PrintCCode(name_list_c)

                    #  ===> Position to repalce IR
                    replace = name_list_c[0]
                    position = -1
                    exists = False
                    for i in range(0, len(name_list_p)):
                        if replace == name_list_p[i]:
                            exists = True
                            position = i

                    # print("Position: ", position,"\n")
                    if position != -1:
                        print("====================> Parent")
                        PrintCCode(root.compute)
                        print("====================> Child")
                        PrintCCode(root.operators[0].compute)
                        root.operators[0], root = move_ir_1(root.operators[0], root, position, parent_op, child_op)
                        print("====================> After fusion")
                        PrintCCode(root.compute)
                else:
                    print("check for N equal")
        
        if (type(root.operators[1]) == TensorOp) and (root.op_type in op_mapping) and (root.operators[1].op_type in op_mapping):
            parent_op = get_operation(root.compute)
            child_op = get_operation(root.operators[1].compute)
            
            if child_op == '+' or child_op == '-' or child_op == '@' or child_op == '/':
                parent_bounds = get_bounds(root.compute[0], [], [], [])
                child_bounds = get_bounds(root.operators[1].compute[0], [], [], [])
                #==================================> For equal bounds
                if parent_bounds == child_bounds:
                    print("==============================================> Safe Right [equal bounds] + , -, /")

                    # ===> Parent
                    body_parent = FindBody(root.compute[0])
                    index_list_parent = get_body_Indices(body_parent)
                    names_parent = get_name_of_all_indices(index_list_parent)
                    name_list_p = names_parent[0]
                    # PrintCCode(name_list_p)

                    # ===> Child
                    body_child = FindBody(root.operators[1].compute[0])
                    index_list_child = get_body_Indices(body_child)
                    names_child = get_name_of_all_indices(index_list_child)
                    name_list_c = names_child[0]
                    # PrintCCode(name_list_c)

                    #  ===> Position to repalce IR
                    replace = name_list_c[0]
                    position = -1
                    exists = False
                    for i in range(0, len(name_list_p)):
                        if replace == name_list_p[i]:
                            exists = True
                            position = i

                    # print("Position: ", position,"\n")
                    if position != -1:
                        print("====================> Parent")
                        PrintCCode(root.compute)
                        print("====================> Child")
                        PrintCCode(root.operators[1].compute)
                        root.operators[1], root = move_ir_1(root.operators[1], root, position, parent_op, child_op)
                        print("====================> After fusion")
                        PrintCCode(root.compute)
                else:
                    print("check for N equal")
    if type(root) == Tensor:
        return
    return root
    
#====================================================================> fuse
def fuse(ast_wit_ir):
    elementwise_op = op_mapping
    node = ast_wit_ir

    print("==============================================> Post Order Traversal")
    node = post_order_traversal(node)
    return node

########################### TEST CASES ###########################
############################################### Simple [+]
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
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("=============================================================================================>")
    print("===============================================================> Updated IR Complete (test 1)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==============================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 1)")
    PrintCCode(new_res_with_ir.compute)

############################################### Simple [+]
def test11():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))
    E = Tensor('e', (30, 30))
    F = Tensor('f', (30, 30))

    res1 = A + B
    res2 = C + D
    res3 = E + F
    res0 = res1 + res2
    res = res0 + res3
    res_with_ir = gen_ir(res)
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("=============================================================================================>")
    print("===============================================================> Updated IR Complete (test 11)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==============================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 11)")
    PrintCCode(new_res_with_ir.compute)

############################################### Simple [-]
def test111():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))
    E = Tensor('e', (30, 30))
    F = Tensor('f', (30, 30))

    res1 = A - B
    res2 = C - D
    res3 = E - F
    res0 = res1 + res2
    res = res0 + res3
    res_with_ir = gen_ir(res)
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("=============================================================================================>")
    print("===============================================================> Updated IR Complete (test 111)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==============================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 111)")
    PrintCCode(new_res_with_ir.compute)

##################################################################################################################################################
def test2():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))

    res1 = A @ B # 20, 40
    res2 = C @ D # 20, 40
    res = res1 + res2
    res_with_ir = gen_ir(res)
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("=============================================================================================>")
    print("===============================================================> Updated IR Complete (test 2)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==============================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 2)")
    PrintCCode(new_res_with_ir.compute)

############################################### Simple [*]
def test22():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))

    res1 = A @ B
    res2 = C + D
    res = res1 + res2
    res_with_ir = gen_ir(res)
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("=============================================================================================>")
    print("===============================================================> Updated IR Complete (test 22)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==============================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 22)")
    PrintCCode(new_res_with_ir.compute)

if __name__ == "__main__":
    test22()