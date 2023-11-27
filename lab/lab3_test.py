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

#====================================================================> PrintCGroupCode
def PrintCGroupCode(index_groups):
    for name, expressions in index_groups.items():
        for expr in expressions:
            print(to_string(expr))

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

#====================================================================> get_body_Indices [[arr14[_l6][_l7], arr10[_l6][_l7],...], [] ]
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

#========================================================================================================================================>> all loop equal
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

    # print("============================>>>>>")
    right_side = body_parts_of_parent[1]
    for j in range(2, len(body_parts_of_parent)):
        oper = ops_seq.pop(0)
        right_side = Expr(right_side, body_parts_of_parent[j], oper)

    new_ass = Assignment(left_side,right_side)
    return new_ass

#====================================================================> update_body ???????????? hard code
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
    if loop_count == 6:
        nested_loop.compute[0].body[0].body[0].body[0].body[0].body[0].body = [new_body]

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

    # print("\n===> loop_iterate_of_parent in reverse")
    loop_iterate_of_parent = get_loop_iterate(tensor_op2.compute[0], [])
    # PrintCCode(loop_iterate_of_parent)

    # print("\n===> loop_iterate_of_child in reverse")
    loop_iterate_of_child = get_loop_iterate(tensor_op1.compute[0], [])
    # PrintCCode(loop_iterate_of_child)

    # print("\n===> Required index_list of parent")
    index_list_p = get_body_Indices(body_of_parent)
    # PrintCCode(index_list_p[0])

    # print("\n===> Required index_list of child")
    index_list = get_body_Indices(body_of_child)
    # PrintCCode(index_list[0])

    # print("\n===> indices_index of child")
    indices_index = get_index_of_all_indices(index_list)
    

    # print("\n===> Chlid Level Info")
    levels_info = get_index_loop_level(loop_iterate_of_child, indices_index)
    # print(levels_info)

    # print("\n===> Replacement")
    updated_index = updated_assignment(index_list[0],loop_iterate_of_parent, levels_info[0])
    # PrintCCode(updated_index)
    replace_this = updated_index[0]
    replacement = updated_index[1]
    
    for i in range(2, len(updated_index)):
        replacement = Expr(replacement, updated_index[i], temp_ops.pop(0))

    tensor_op1.compute = []
    # print("\n===> Updated Assignment of Parent")
    # print("==========> Position: ", position)
    new_body_of_2 = new_body_a(index_list_p[0], body_of_parent,replacement, parent_op, child_op, replace_this)
    # print(to_string(new_body_of_2))
    
    # print("=============================> Here")
    tensor_op2 = update_body_a(tensor_op2, new_body_of_2)
    
    return (tensor_op1, tensor_op2)

#================================================================================================================================================>
#=========================================================== Functions for @ case ===============================================================>
#================================================================================================================================================>
#====================================================> update_child_body
def update_child_body(child_body, to_insert, position):
    if position == 0:
        child_body.append(to_insert[0])
        return
    
    if type(child_body) == Loop:
        update_child_body(child_body.body, to_insert, position-1)
    else:
        update_child_body(child_body[0].body, to_insert, position-1)
    return child_body

#====================================================> updated_itrator
def updated_assignment_mul(index_list,loop_iterate_of_child,levels_info, matching_loops_count, loop_count):
    ret_index_list = []
    
    for i  in range(0, len(index_list)):
        ind = index_list[i]
        def recursive_index_update_mul(ind, loop_iterate_of_child, levels_info, matching_loops_count, loop_count):
            if loop_count == 0:
                return ind, matching_loops_count
            
            ind.dobject, matching_loops_count = recursive_index_update_mul(ind.dobject, loop_iterate_of_child, levels_info, matching_loops_count, loop_count-1)
            if matching_loops_count > 0:
                ind.index = loop_iterate_of_child[levels_info.pop(0)]
                matching_loops_count = matching_loops_count -1
            return ind, matching_loops_count
        
        ind, _ = recursive_index_update_mul(ind, loop_iterate_of_child, levels_info[i], matching_loops_count, loop_count)
        ret_index_list.append(ind)
    return ret_index_list

def replace_parent_body(parent, new_body):
    if not type(parent) == Loop:
        parent = [new_body]
        return
    if type(parent.body[0]) == Loop:
        return replace_parent_body(parent.body[0], new_body)
    else:
        parent.body = [new_body]
        return
    
##################################################################################################################################################
######################################################### post_order_traversal FUNCTION ##########################################################
##################################################################################################################################################
def post_order_traversal(root):
    if type(root) == TensorOp:
        post_order_traversal(root.operators[0])
        post_order_traversal(root.operators[1])
        #====================================================================================================> Left Child Check
        if (type(root.operators[0]) == TensorOp):
            #====================================================================================================> scaler +, -, /, *
            if ((root.op_type in op_mapping) and (root.operators[0].op_type in op_mapping)):
                parent_bounds = get_bounds(root.compute[0], [], [], [])
                child_bounds = get_bounds(root.operators[0].compute[0], [], [], [])

                parent_op = get_operation(root.compute)
                child_op = get_operation(root.operators[0].compute)
                #======================================================================> For equal bounds
                if parent_bounds == child_bounds:
                    print("==============================================> Safe Left Fuse [equal bounds] + , -, /")

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
                        #################################################################### Important prints
                        print("====================> Parent")
                        PrintCCode(root.compute)
                        print("====================> Child")
                        PrintCCode(root.operators[0].compute)
                        root.operators[0], root = move_ir_1(root.operators[0], root, position, parent_op, child_op)
                        print("====================> After fusion")
                        PrintCCode(root.compute)
                else:
                    print("check for N equal?????????????????????")

            #====================================================================================================> einsum Left
            if (root.operators[0].op_type == "einsum"  and root.op_type != "einsum"):
                # print("======> in einsum")
                parent_bounds = get_bounds(root.compute[0], [], [], [])
                child_bounds = get_bounds(root.operators[0].compute[0], [], [], [])
                # print("===> Parent/Child Bounds")
                # print(parent_bounds)
                # print(child_bounds, "\n")

                equal_loops = 0
                for a in range(0, len(parent_bounds[0])):
                    if (parent_bounds[0][a] == child_bounds[0][a]) and (parent_bounds[1][a] == child_bounds[1][a]) and (parent_bounds[2][a] == child_bounds[2][a]):
                        equal_loops = equal_loops + 1
                # print("N equal loop bounds: ",equal_loops,"\n")
                
                if equal_loops != 0:
                    #====================================================> For getting the updated parent body assignment
                    body_of_child = FindBody(root.operators[0].compute[0])
                    body_of_parent = FindBody(root.compute[0])

                    temp = to_string(body_of_parent[0])
                    temp_ops = re.findall(r'[\+\-\*/]', temp)

                    loop_iterate_of_child = get_loop_iterate(root.operators[0].compute[0], [])
                    loop_iterate_of_parent = get_loop_iterate(root.compute[0], [])

                    index_list_cc = get_body_Indices(body_of_child)
                    index_list_pp = get_body_Indices(body_of_parent)


                    indices_index = get_index_of_all_indices(index_list_pp)
                    levels_info22 = get_index_loop_level(loop_iterate_of_parent, indices_index)

                    count = count_loops(root.compute[0], 0)

                    updated_index = updated_assignment_mul(index_list_pp[0],loop_iterate_of_child, levels_info22[0], equal_loops, count)

                    new_rhs = updated_index[1]
                    for i in range(2, len(updated_index)):
                        new_rhs = Expr(new_rhs, updated_index[i], temp_ops.pop(0))

                    new_assignment = Assignment(updated_index[0], new_rhs)

                    replace_parent_body(root.compute[0], new_assignment)
                    # ====================================================>
                    to_insert = root.compute[0]
                    for b in range(0, equal_loops):
                        # print(type(to_insert))
                        if(type(to_insert) == Loop):
                            to_insert = to_insert.body
                        else:
                            to_insert = to_insert[0].body

                    child_body = update_child_body(root.operators[0].compute[0], to_insert,equal_loops)
                    
                    root.compute = [child_body]
                    root.operators[0].compute = []
                
        #====================================================================================================> Right Child Check
        if (type(root.operators[1]) == TensorOp):
            #====================================================================================================> scaler +, -, /
            if ((root.op_type in op_mapping) and (root.operators[1].op_type in op_mapping)):
                parent_bounds = get_bounds(root.compute[0], [], [], [])
                child_bounds = get_bounds(root.operators[1].compute[0], [], [], [])

                parent_op = get_operation(root.compute)
                child_op = get_operation(root.operators[1].compute)
                #================================================================================================> For equal bounds
                if parent_bounds == child_bounds:
                    print("==============================================> Safe Right Fuse [equal bounds] + , -, /")
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
                        #################################################################### Important prints
                        print("====================> Parent")
                        PrintCCode(root.compute)
                        print("====================> Child")
                        PrintCCode(root.operators[1].compute)
                        root.operators[1], root = move_ir_1(root.operators[1], root, position, parent_op, child_op)
                        print("====================> After fusion")
                        PrintCCode(root.compute)
                else:
                    print("check for N equal ????????????????????????")
    
    if type(root) == Tensor:
        return
    
    return root

##################################################################################################################################################
############################################################### FUSE FUNCTION ####################################################################
##################################################################################################################################################
def fuse(ast_wit_ir):
    node = ast_wit_ir
    print("=========================================> Post Order Traversal")
    node = post_order_traversal(node)
    return node

##################################################################################################################################################
############################################################### TEST CASES #######################################################################
##################################################################################################################################################

################################################################################################################################################## Simple [+]
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
    
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 1)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 1)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Simple [+]
def test2():
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
    
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 2)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 2)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Simple [- and /]
def test3():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))
    E = Tensor('e', (30, 30))
    F = Tensor('f', (30, 30))

    res1 = A - B
    res2 = C // D
    res3 = E - F
    res0 = res1 + res2
    res = res0 + res3
    res_with_ir = gen_ir(res)
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 3)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 3)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Not all loops bound are equal Simple [+]
def test4():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))

    res1 = A + B
    res2 = C * D

    res = res1 + res2
    res_with_ir = gen_ir(res)
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 4)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 4)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## From Notion
def test5():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (20, 30))
    C = Tensor('c', (10, 30))

    res1 = A @ B # 20, 40
    res = res1 + C
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 5)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 5)")
    PrintCCode(new_res_with_ir.compute)

##################################################################################################################################################  From Notion Equal Bounds
def test6():
    A = Tensor('a', (20, 20))
    B = Tensor('b', (20, 20))
    C = Tensor('c', (20, 20))

    res1 = A @ B # 20, 40
    res = res1 + C
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 6)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 6)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Simple [*]
def test7():
    A = Tensor('a', (20, 20))
    B = Tensor('b', (20, 20))
    C = Tensor('c', (20, 20))

    res1 = A * B # 20, 40
    res = res1 + C
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 7)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 7)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Simple [-]
def test8():
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
    
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 8)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 8)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Simple [/]
def test9():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))
    E = Tensor('e', (30, 30))
    F = Tensor('f', (30, 30))

    res1 = A // B
    res2 = C // D
    res3 = E // F
    res0 = res1 + res2
    res = res0 + res3
    res_with_ir = gen_ir(res)
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 9)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 9)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Simple [*]
def test10():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))
    E = Tensor('e', (30, 30))
    F = Tensor('f', (30, 30))

    res1 = A * B
    res2 = C * D
    res3 = E * F
    res0 = res1 + res2
    res = res0 + res3
    res_with_ir = gen_ir(res)
    code0 = codegen.cpu.gen_cpp(res_with_ir)
    
    print("===============================================================> res_with_ir [IR]")
    print(code0)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)
    
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 10)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 10)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Scalar Operation
def test11():
    A = Tensor('a', (30, 30))
    scalar = 5

    res = A + scalar
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)

    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 11)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 11)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Broadcasting Case
def test12():
    A = Tensor('a', (30, 1))
    B = Tensor('b', (1, 30))

    res = A + B
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)

    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 12)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 12)")
    PrintCCode(new_res_with_ir.compute)

################################################################################################################################################## Reduction Operation
def test13():
    A = Tensor('a', (30, 30))

    res = A.sum(axis=0)
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)

    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 13)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 13)")
    PrintCCode(new_res_with_ir.compute)

##################################################################################################################################################
def test14():
    A = Tensor('a', (20, 30))
    B = Tensor('b', (30, 40))
    C = Tensor('c', (20, 40))
    D = Tensor('d', (40, 40))

    res1 = A + B # 20, 40
    res2 = C + D # 20, 40
    res = res1 + res2
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 14)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 14)")
    PrintCCode(new_res_with_ir.compute)

##################################################################################################################################################
def test15():
    A = Tensor('a', (20, 30))
    B = Tensor('b', (30, 40))
    C = Tensor('c', (20, 40))
    D = Tensor('d', (40, 40))

    res1 = A // B # 20, 40
    res2 = C + D # 20, 40
    res = res1 + res2
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 15)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 15)")
    PrintCCode(new_res_with_ir.compute)

##################################################################################################################################################
def test16():
    A = Tensor('a', (20, 30))
    B = Tensor('b', (30, 40))
    C = Tensor('c', (20, 40))
    D = Tensor('d', (40, 40))

    res1 = A // B # 20, 40
    res2 = C * D # 20, 40
    res = res1 + res2
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 16)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 16)")
    PrintCCode(new_res_with_ir.compute)

##################################################################################################################################################
def test17():
    A = Tensor('a', (20, 30))
    B = Tensor('b', (30, 40))
    C = Tensor('c', (20, 40))
    D = Tensor('d', (40, 40))

    res1 = A @ B # 20, 40
    res2 = C @ D # 20, 40
    res = res1 + res2
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print("===============================================================> res_with_ir [IR]")
    print(code)
    print("===============================================================>")
    new_res_with_ir = fuse(res_with_ir)

    print("==========================================================================================================>")
    print("===============================================================> Updated IR Complete (test 17)")
    code1 = codegen.cpu.gen_cpp(new_res_with_ir)
    print(code1)
    print("==========================================================================================================>")
    print("===============================================================> Updated IR Loops Only (test 17)")
    PrintCCode(new_res_with_ir.compute)

if __name__ == "__main__":
    test15()