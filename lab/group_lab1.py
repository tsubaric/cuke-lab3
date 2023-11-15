import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codegen.cpu import to_string, gen_cpp  # noqa: E402 <- disables linter warning
from core.ir import Scalar, Ndarray, Loop, Assignment, Expr, Index, Decl  # noqa: E402 <- disables linter warning

def PrintCCode(ir):
    code = ''
    for d in ir:
        if d:
            code += to_string(d)
    print(code)

def PrintCGroupCode(index_groups):
    for name, expressions in index_groups.items():
        for expr in expressions:
            print(to_string(expr))

def Loop0():
    ir = []

    N = Scalar('int', 'N')
    M = Scalar('int', 'M')
    L = Scalar('int', 'L')
    A = Ndarray('int', (N, M, L), 'A')
    B = Ndarray('int', (N, M, L), 'B')

    loopi = Loop(0, N, 1, [])
    loopj = Loop(0, M, 1, [])
    loopk = Loop(0, L, 1, [])

    loopi.body.append(loopj)
    loopj.body.append(loopk)

    lhs1 = Index(Index(Index(A, Expr(loopi.iterate, 1, '+')),
                 loopj.iterate), loopk.iterate)
    lhs2 = Index(Index(Index(B, Expr(loopi.iterate, 1, '+')),
                 Expr(loopj.iterate, 2, '+')), Expr(loopk.iterate, 1, '-'))
    rhs1 = Index(Index(Index(B, Expr(loopi.iterate, 1, '+')),
                 loopj.iterate), Expr(loopk.iterate, 1, '-'))
    rhs2 = Index(Index(Index(A, loopi.iterate), loopj.iterate),
                 Expr(loopk.iterate, 1, '+'))
    rhs3 = Index(Index(Index(B, loopi.iterate), Expr(
        loopj.iterate, 2, '+')),  loopk.iterate)

    # body = Assignment(lhs, Expr(rhs1, rhs2, '+'))
    loopk.body.extend([Assignment(lhs1, Expr(rhs1, 2, '+')),
                      Assignment(lhs2, Expr(rhs2, rhs3, '+'))])

    ir.extend([Decl(L)])
    ir.extend([Decl(M)])
    ir.extend([Decl(N)])
    ir.extend([Decl(A)])
    ir.extend([loopi])

    return ir

# for ( k = 0; k < L ; ++k ){
# 	for ( j = 0; j < M; ++ j ){
# 		for ( i = 0; i < N; ++ i ){
# 			a[i+1] [j+1] [k] = a [i] [j] [k] + a [i] [j + 1] [k + 1] ;
# 		}
# 	}
# }

# Distance Vector:
# [1, 1, 0] :  a[i+1] [j+1] [k] and a [i] [j] [k]
# [1, 0, -1] : a[i+1] [j+1] [k] and  a [i] [j + 1] [k + 1]

# Direction Vector:
# [<, <, =]
# [<, =, >]

def Loop1():
    ir = []

    L = Scalar('int', 'L')
    M = Scalar('int', 'M')
    N = Scalar('int', 'N')
    A = Ndarray('int', (N, M, L), 'A')

    loopk = Loop(0, L, 1, [])
    loopj = Loop(0, M, 1, [])
    loopi = Loop(0, N, 1, [])
    loopk.body.append(loopj)
    loopj.body.append(loopi)

    lhs = Index(Index(Index(A, Expr(loopi.iterate, 1, '+')),
                Expr(loopj.iterate, 1, '+')), loopk.iterate)
    rhs1 = Index(Index(Index(A, loopi.iterate), loopj.iterate), loopk.iterate)
    rhs2 = Index(Index(Index(A, loopi.iterate), Expr(
        loopj.iterate, 1, '+')),  Expr(loopk.iterate, 1, '+'))

    body = Assignment(lhs, Expr(rhs1, rhs2, '+'))
    loopi.body.append(body)

    ir.extend([Decl(L)])
    ir.extend([Decl(M)])
    ir.extend([Decl(N)])
    ir.extend([Decl(A)])
    ir.extend([loopk])

    return ir

# for ( i = 0; i < N ; ++i ){
# 	for ( j = 0; j < N; ++ j ){
# 			a[i][j] = a[i+1][j-1];
# 	}
# }

# Distance Vector:
# [-1, 1]

# Direction Vector:
# [<, >]

def Loop2():
    ir = []

    N = Scalar('int', 'N')
    A = Ndarray('int', (N, N), 'A')

    loopi = Loop(0, N, 1, [])
    loopj = Loop(0, N, 1, [])

    loopi.body.append(loopj)

    lhs = Index(Index(A, loopi.iterate), loopj.iterate)
    rhs = Index(Index(A, Expr(loopi.iterate, 1, '+')),
                Expr(loopj.iterate, 1, '-'))

    loopj.body.append(Assignment(lhs, rhs))

    ir.extend([Decl(N)])
    ir.extend([Decl(A)])
    ir.extend([loopi])

    return ir

# 5. Safety checking based on the direction vector we get:
# Exchange [0, 1]
# [=, <, >]
# [<, =, =]
# [=, <, >]

# Exchange [0, 2]
# [>, =, <] The first

def is_interchange_safe(direction_vectors, loop_indices_to_interchange):
    # Loop through the direction vectors, checking if the interchange changes
    # the semantics of the program
    for vector in direction_vectors:
        for loop_idx in range(loop_indices_to_interchange[0], loop_indices_to_interchange[1]):
            if (vector[loop_idx] != '=' and
                vector[loop_indices_to_interchange[1]] != '=' and
                    vector[loop_idx] != vector[loop_indices_to_interchange[1]]):
                print('False')
                return False

    return True

# 4. Compute the direction vector
# [Write expression, Read expression]
# [A[_l10 + 1][_l1][_l2], A[_l0][_l1][_l2 + 1]]
# [B[_l0 + 1][_l1 + 2][_l2 - 1], B[_l0 + 1][_l1][_l2 - 1]]
# [B[_l0 + 1][_l1 + 2][_l2 - 1], B[_l0][_l1 + 2][_l2]]

# Distance vector:
# [1, 0, -1]
# [0, 2,  0]
# [1, 0, -1]

# Direction vector
# [<, =, >]
# [=, <, =]
# [<, =, >]

def direction_vector(dist_vect):
    dir_vect = []
    for i in range(0, len(dist_vect)):
        directions = []
        for j in range(0, len(dist_vect[i])):
            if dist_vect[i][j] > 0:
                directions.append('<')
            elif dist_vect[i][j] < 0:
                directions.append('>')
            else:
                directions.append('=')
        dir_vect.append(directions)
    return dir_vect

# Helper functeion for extracting int from i.e "_l0 + 1"

def extract_integer_value(exp):
    exp = exp.replace(" ", "")
    for character in range(0, len(exp)):
        if len(exp) > 1:
            if exp[character].isdigit() and exp[character-1].isalpha():
                pass
            elif exp[character].isdigit() and exp[character-1] == '-':
                return -1 * int(exp[character])
            elif exp[character].isdigit():
                return int(exp[character])
    return 0

# calculate the distance but in reverse and then correcting

def distance_vector(combinations):
    def calculate_distance(combo, dist_vect, temp_dist):
        if type(combo[0]) == Ndarray and type(combo[1]) == Ndarray:
            return dist_vect.append(temp_dist)

        elif type(combo[0]) == Index and type(combo[1]) == Index:
            index_1 = to_string(combo[0].index)
            index_2 = to_string(combo[1].index)

            numeric_value1 = extract_integer_value(index_1)
            numeric_value2 = extract_integer_value(index_2)
            distance = numeric_value1 - numeric_value2

            temp_dist.append(distance)
            calculate_distance(
                [combo[0].dobject, combo[1].dobject], dist_vect, temp_dist)

    dist_vect = []
    for combo in combinations:
        temp_dist = []
        dist = calculate_distance(combo, dist_vect, temp_dist)
        dist_vect.append(dist)
    filtered_dist_vect = [x for x in dist_vect if x is not None]
    final_dist = []
    for vec in filtered_dist_vect:
        vec = list(reversed(vec))
        final_dist.append(vec)

    return final_dist

def Write_expression_Read_expression_combination(write_dic, read_dic):
    combinations = []
    for key in write_dic.keys():
        for writer in range(0, len(write_dic[key])):
            key_writer_list = write_dic[key]
            key_reader_list = read_dic[key]
            for reader in range(0, len(key_reader_list)):
                combinations.append(
                    [key_writer_list[writer], key_reader_list[reader]])

    return combinations

# 3. Group the index statement by there names.
# Two dict:
#     Write dicts: {A : [A[_l10 + 1][_l1][_l2]], 'B' : [B[_l0 + 1][_l1 + 2][_l2 - 1]]}
#     Read dicts:  {A : [A[_l0][_l1][_l2 + 1]], 'B' : [B[_l0 + 1][_l1][_l2 - 1], B[_l0][_l1 + 2][_l2]]}

def read_write_dic(write_expr, read_expr):
    def recursive_group(original_statement, cur_statement,  res_dict):
        if type(cur_statement) == Ndarray:
            name = cur_statement.__name__
            if (name not in res_dict):
                res_dict[name] = [original_statement]
            else:
                res_dict[name].append(original_statement)

        elif type(cur_statement) == Index:
            recursive_group(original_statement,
                            cur_statement.dobject, res_dict)

    write_index_groups = {}
    read_index_groups = {}
    for statement in write_expr:
        recursive_group(statement, statement, write_index_groups)
    for statement in read_expr:
        recursive_group(statement, statement, read_index_groups)
    return write_index_groups, read_index_groups

# 2. Get the index statement of the loop body
# write ststement array: A[_l0 + 1][_l1][_l2]B[_l0 + 1][_l1 + 2][_l2 - 1]
# read statement array: B[_l0 + 1][_l1][_l2 - 1]A[_l0][_l1][_l2 + 1]B[_l0][_l1 + 2][_l2]

def GetIndex(statement, is_write, write_expr=[], read_expr=[]):
    if type(statement) == Ndarray or type(statement) == Index:
        if is_write:
            write_expr.append(statement)
        else:
            read_expr.append(statement)

    if type(statement) == Assignment:
        GetIndex(statement.lhs, True, write_expr, read_expr)
        GetIndex(statement.rhs, False, write_expr, read_expr)
    elif type(statement) == Expr:
        GetIndex(statement.left, is_write, write_expr, read_expr)
        GetIndex(statement.right, is_write, write_expr, read_expr)
    else:
        return

# 1. Identify the loop body
# the output of first step
# A[_l0 + 1][_l1][_l2] = B[_l0 + 1][_l1][_l2 - 1] + 2;
# B[_l0 + 1][_l1 + 2][_l2 - 1] = A[_l0][_l1][_l2 + 1] + B[_l0][_l1 + 2][_l2];

def FindBody(nested_loop):
    if not type(nested_loop) == Loop:
        return nested_loop
    if type(nested_loop.body[0]) == Loop:
        return FindBody(nested_loop.body[0])
    else:
        return nested_loop.body

# will swap loops regardless of semantic safety
# intended to be called only after safety check
def SwapLoops(ir, loop_idx):
    ir = copy.deepcopy(ir)
    # if the first swap loop index is zero we need to know where 
    # to put the inner loop in the ir
    loop_replace_index = 0

    for index, ir_item in enumerate(ir):
        if type(ir_item) == Loop:
            outer_loop = ir_item
            loop_replace_index = index
            break
    outer_loop_parent = None
    # recursively find the outer loop of the exchange
    # store the parent of the outer loop to later make the inner loop
    # a child of the parent/child of the outer loop to swap later
    for _ in range(0, loop_idx[0]):
        if type(outer_loop.body[0]) == Loop:
            outer_loop_parent = outer_loop
            outer_loop = outer_loop.body[0]
        else:
            raise Exception("Loop index out of bounds")
    outer_loop_body = outer_loop.body

    # recursively find the inner loop of the exchange
    # store the parent/child of the inner loop to swap later
    inner_loop = outer_loop.body[0]
    inner_loop_parent = outer_loop
    for _ in range(loop_idx[0], loop_idx[1]-1):
        if type(inner_loop.body[0]) == Loop:
            inner_loop_parent = inner_loop
            inner_loop = inner_loop.body[0]
        else:
            raise Exception("Loop index out of bounds")
    inner_loop_body = inner_loop.body

    # swap the loops
    # print(outer_loop)
    # print(inner_loop)
    if outer_loop_parent is None:
        ir[loop_replace_index] = inner_loop
    else:
        outer_loop_parent.body[0] = inner_loop
    inner_loop_parent.body[0] = outer_loop
    outer_loop.body = inner_loop_body
    inner_loop.body = outer_loop_body

    return ir
        
def InterchangeLoop(ir, loop_idx=[]):
    write_expr = []
    read_expr = []
    for ir_item in ir:
        if type(ir_item) == Loop:
            body = FindBody(ir_item)
            for body_item in body:
                GetIndex(body_item, False, write_expr, read_expr)

    write_index_groups, read_index_groups = read_write_dic(
        write_expr, read_expr)
    returned_combination = Write_expression_Read_expression_combination(
        write_index_groups, read_index_groups)
    dist_vec = distance_vector(returned_combination)
    dir_vect = direction_vector(dist_vec)
    is_safe = is_interchange_safe(dir_vect, loop_idx)

    # Testing Each Step Output
    print("<===== (Step 1) Loop body we got! =====>")
    PrintCCode(body)

    print("<===== (Step 2) Read/Write Statements =====>")
    PrintCCode(write_expr)
    PrintCCode(read_expr)

    print("<===== (Step 3) Write Dic! =====>")
    PrintCGroupCode(write_index_groups)

    print("\n<=====  (Step 3) Read Dic! =====>")
    PrintCGroupCode(read_index_groups)

    print("<===== (Step 4) Write/Read combinations =====>")
    for comb in range(0, len(returned_combination)):
        PrintCCode(returned_combination[comb])

    print("\n<===== (Step 4) Distance vector =====>")
    print(dist_vec)

    print("\n<===== (Step 4) Direction vector =====>")
    print(dir_vect)

    print("\n<===== (Step 5) Interchange Safety Check =====>")
    if is_safe:
        print("Loop interchange is safe. Direction vectors do not indicate conflicts.")
    else:
        print("Loop interchange is not safe. Conflicting directions detected in direction vectors.")

    if not is_safe:
        return False, ir

    ir_res = SwapLoops(ir, loop_idx)
    return True, ir_res

if __name__ == "__main__":
    loop0_ir = Loop0()
    loop1_ir = Loop1()
    loop2_ir = Loop2()
    PrintCCode(loop0_ir)

    swapped0, optimized_loop0_ir = InterchangeLoop(loop0_ir, [0, 1])
    # swapped1, optimized_loop1_ir = InterchangeLoop(loop1_ir, [1, 2])
    # swapped2, optimized_loop2_ir = InterchangeLoop(loop2_ir, [0, 1])

    PrintCCode(loop0_ir)
    PrintCCode(optimized_loop0_ir)

    # optimized_ir = LoopInterchange(ir)
    # print("Loop after interchange:")
    # PrintCCode(optimized_ir)