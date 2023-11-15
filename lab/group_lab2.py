import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ir import *
from codegen.cpu import *

def PrintCCode(ir):
	code = ''
	for d in ir:
		# if d:
			code += to_string(d)
	print(code,'\n')

#===========================================================>
# for (int _l0 = 0; _l0 < 20; _l0 += 1) {
# for (int _l1 = _l0 + 3; _l1 < _l0 + 10; _l1 += 1) {
# for (int _l2 = _l1 + 20; _l2 < _l1 + 30; _l2 += 1) {
# A[_l0 + 1][_l1][_l2] = B[_l0][_l1][_l2] + 2;
# } 
# } 
# }
#===========================================================>
def Loop0():
    ir = []

    L = Scalar('int', 'L')
    M = Scalar('int', 'M')
    N = Scalar('int', 'N')
    A = Ndarray('int', (N, M, L), 'A')
    B = Ndarray('int', (N, M, L), 'B')

    loopi = Loop(0, 20, 1, [])
    loopj = Loop(Expr(loopi.iterate, 3, '+'),  Expr(loopi.iterate, 10, '+'), 1, [])
    loopk = Loop(Expr(loopj.iterate, 20, '+'), Expr(loopj.iterate, 30, '+'), 1, [])

    loopi.body.append(loopj)
    loopj.body.append(loopk)

    lhs1 = Index(Index(Index(A, Expr(loopi.iterate, 1, '+')), loopj.iterate), loopk.iterate)
    rhs1 = Index(Index(Index(B, loopi.iterate), loopj.iterate), loopk.iterate)
	
    # body = Assignment(lhs, Expr(rhs1, rhs2, '+'))
    loopk.body.extend([Assignment(lhs1, Expr(rhs1, 2, '+'))])

    ir.extend([Decl(L)])
    ir.extend([Decl(M)])
    ir.extend([Decl(N)])
    ir.extend([Decl(A)])
    ir.extend([Decl(B)])
    ir.extend([loopi])

    return ir

#===========================================================>
# for (int _l3 = 0; _l3 < L; _l3 += 1) {
# for (int _l4 = 0; _l4 < M; _l4 += 1) {
# for (int _l5 = 0; _l5 < N; _l5 += 1) {
# A[_l5 + 1][_l4 + 1][_l3] = A[_l5][_l4][_l3] + A[_l5][_l4 + 1][_l3 + 1];
# } 
# } 
# } 
#===========================================================>
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

#===========================================================>
# for (int _l6 = 0; _l6 < N; _l6 += 1) {
# for (int _l7 = 0; _l7 < N; _l7 += 1) {
# A[_l6][_l7] = A[_l6 + 1][_l7 - 1];
# } 
# } 
#===========================================================>
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

#===========================================================>
#===========================================================>
#===========================================================> GetKeyInfo
def GetKeyInfo(loop_ir):
    def _GetKeyInfo(loop_ir, lower_bounds, upper_bounds, index_dict, level):
        if not type(loop_ir)==Loop:
            return
        if type(loop_ir)==Loop:
            lower_bounds.append(loop_ir.start)
            upper_bounds.append(loop_ir.end)
            index_dict[loop_ir.iterate] = level
            _GetKeyInfo(loop_ir.body[0], lower_bounds, upper_bounds, index_dict, level+1)

    index_dict = {}
    lower_bounds = []
    upper_bounds = []
    _GetKeyInfo(loop_ir, lower_bounds, upper_bounds, index_dict, 0)
    return lower_bounds, upper_bounds, index_dict


#===========================================================>
#===========================================================>
#===========================================================> SetKeyInfo
def SetKeyInfo(loop_ir, low_bounds, up_bounds, tile_size):
    def _SetKeyInfo(loop_ir, low_bounds, up_bounds, level, tile_size):
        if not type(loop_ir)==Loop:
            return
        if type(loop_ir)==Loop:
            loop_ir.start = low_bounds[level]
            loop_ir.end = up_bounds[level]
            loop_ir.step = tile_size[level]
            _SetKeyInfo(loop_ir.body[0], low_bounds, up_bounds, level+1, tile_size)
    
    _SetKeyInfo(loop_ir, low_bounds, up_bounds, 0, tile_size)
    return loop_ir


#===========================================================>
#===========================================================>
#===========================================================> GetNewUpperBound
def GetNewUpperBound(upper_bound_expr, index_dict, tile_size_list):
    if type(upper_bound_expr)==Expr: # upper_bound_expr.left + upper_bound_expr.right 
        iterator_index = upper_bound_expr.left
        return Expr(upper_bound_expr, Expr(tile_size_list[index_dict[iterator_index]],1,'-'), '+')
    else:
        return upper_bound_expr


#===========================================================>
#===========================================================>
#===========================================================> GetNewLowerBound
def GetNewLowerBound(lower_bound_expr, tile_size,i):
    # New lower bound(i) = Original lower bound(i) - (tile_size(i) - 1)
    return Expr(lower_bound_expr, Expr(tile_size[i], 1, '-'), '-')


#===========================================================>
#===========================================================>
#===========================================================> FindBody
def FindBody(nested_loop):
    if not type(nested_loop) == Loop:
        return nested_loop
    if type(nested_loop.body[0]) == Loop:
        return FindBody(nested_loop.body[0])
    else:
        return nested_loop.body


#===========================================================>
#===========================================================>
#===========================================================> point_loops generation
def point_loops(org_low, org_up, index_dict, tile_size, num_of_loops,bodyyy):
    point_ir = []
    loops = [0 for _ in range(num_of_loops)]
    tile_loop_iter = list(index_dict.keys())
    
    for kk in range(0,num_of_loops):
        loops[kk] = Loop(Max(tile_loop_iter[kk],org_low[kk]), Min(Expr(tile_loop_iter[kk],tile_size[kk],'+'),org_up[kk]),1,[])
      
    for jj in range(num_of_loops-1):
        loops[jj].body.append(loops[jj+1])  
    
    loops[-1].body.extend(bodyyy)
    point_ir.extend([loops[0]])
    
    return point_ir


#===========================================================>
#===========================================================>
#===========================================================> Merge Tiled and Point Loops
def merge(tiled_loops, point_loops):
    if not type(tiled_loops) == Loop:
        tiled_loops = point_loops
    if type(tiled_loops.body[0]) == Loop:
        return merge(tiled_loops.body[0], point_loops)
    else:
        tiled_loops.body = point_loops

   
#===========================================================>
#===========================================================>
#===========================================================> LoopTiling
def LoopTiling(ir, tile_size = []):
    new_ir = ir
    low_bounds = []
    up_bounds = []
    body = ""
    #====================================> lower_bounds, upper_bounds change
    for ir_item in new_ir:
        if type(ir_item) == Loop:
            org_lower_bounds, org_upper_bounds, index_dict = GetKeyInfo(ir_item)
            i = 0
            for lower_bound_expr in org_lower_bounds:
                new_lower_bound = GetNewLowerBound(lower_bound_expr, tile_size,i)
                low_bounds.append(new_lower_bound)
                i = i + 1

            for upper_bound_expr in org_upper_bounds:
                #Type(upper_bound_expr) is an Exper or a nunmber
                new_upper_bound = GetNewUpperBound(upper_bound_expr, index_dict, tile_size)
                up_bounds.append(new_upper_bound)
            
            ir_item = SetKeyInfo(ir_item, low_bounds, up_bounds, tile_size)
            body = FindBody(ir_item)
    
    point_loops_ir = point_loops(org_lower_bounds,org_upper_bounds,index_dict, tile_size, len(low_bounds),body)
    
    for ir_item in new_ir:
        if type(ir_item) == Loop:
            merge(ir_item,point_loops_ir)
       
    # ========================================================================> Printing
    # print("============================> New Lower Bounds")
    # PrintCCode(low_bounds)
    
    # print("============================> New Upper Bounds")
    # PrintCCode(up_bounds)
    
    # print("============================> New Points Loops Generated")
    # PrintCCode(point_loops_ir)  
    
    # print("============================> Inner Most Statements")
    # PrintCCode(body)
    
    # print("============================> Merged Final Loops")
    # PrintCCode(ir)
    # ========================================================================>
    return new_ir
            
	
if __name__ == "__main__":
    print("============================> Original code")
    loop0_ir = Loop0()  
    loop1_ir = Loop1()
    loop2_ir = Loop2()
    PrintCCode(loop0_ir)

    print("============================> Updated code")
    returned_ir = LoopTiling(loop0_ir, tile_size = [3,4,5])
    PrintCCode(returned_ir)