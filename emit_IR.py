from typing import Dict, Iterator, List, Optional, cast

import bytecode
import sexp
from errors import EnvBindingNotFound
from visitor import Visitor


class FunctionEmitter(Visitor):
    def __init__(self, global_env: Dict[scheme.SSym, scheme.Value]) -> None:
        self.global_env = global_env

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        local_env: Dict[sexp.SSym, bytecode.Var] = {}
        for param in func.params:
            local_env[param] = bytecode.Var(param.name)

        bb_names = name_generator('bb')
        var_names = name_generator('var')

        body_exprs = list(func.body)

        assert len(body_exprs) > 0, 'Function bodies must not be empty'

        start_block = bytecode.BasicBlock(next(bb_names))
        emitter = ExpressionEmitter(
            start_block, bb_names, var_names, local_env, self.global_env)
        for expr in body_exprs[:-1]:
            emitter.visit(expr)
            emitter = ExpressionEmitter(
                emitter.end_block, bb_names, var_names,
                local_env, self.global_env)

        emitter.visit(body_exprs[-1])

        return_instr = bytecode.ReturnInst(emitter.result)
        emitter.end_block.add_inst(return_instr)

        emitted_func = bytecode.Function(
            [bytecode.Var(param.name) for param in func.params],
            start_block
        )

        func.code = emitted_func
        self.global_env[func.name] = func


class ExpressionEmitter(Visitor):
    result: bytecode.Parameter

    def __init__(self, parent_block: bytecode.BasicBlock,
                 bb_names: Iterator[str],
                 var_names: Iterator[str],
                 local_env: Dict[scheme.SSym, bytecode.Var],
                 global_env: Dict[scheme.SSym, scheme.Value],
                 quoted: bool = False) -> None:
        self.parent_block = parent_block
        self.end_block = parent_block
        self.bb_names = bb_names
        self.var_names = var_names
        self.local_env = local_env
        self.global_env = global_env
        self.quoted = quoted

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        assert func.is_lambda, 'Nested named functions not supported'
        assert not self.quoted, 'Non-primitives in quoted list unsupported'

        func_emitter = FunctionEmitter(self.global_env)
        func_emitter.visit(func)

        lambda_var = bytecode.Var(next(self.var_names))
        lookup_lambda_instr = bytecode.LookupInst(
            lambda_var, bytecode.SymLit(func.name))

        self.parent_block.add_inst(lookup_lambda_instr)
        self.result = lambda_var

    def visit_SConditional(self, conditional: sexp.SConditional) -> None:
        assert not self.quoted, 'Non-primitives in quoted list unsupported'

        test_emitter = ExpressionEmitter(
            self.parent_block, self.bb_names, self.var_names,
            self.local_env, self.global_env)
        test_emitter.visit(conditional.test)

        then_block = bytecode.BasicBlock(next(self.bb_names))
        else_block = bytecode.BasicBlock(next(self.bb_names))

        result_var = bytecode.Var(next(self.var_names))

        then_br_instr = bytecode.BrInst(test_emitter.result, then_block)
        else_br_instr = bytecode.JmpInst(else_block)

        test_emitter.end_block.add_inst(then_br_instr)
        test_emitter.end_block.add_inst(else_br_instr)

        then_emitter = ExpressionEmitter(
            then_block, self.bb_names, self.var_names,
            self.local_env, self.global_env)
        then_emitter.visit(conditional.then_expr)

        then_result_instr = bytecode.CopyInst(result_var, then_emitter.result)
        then_emitter.end_block.add_inst(then_result_instr)

        else_emitter = ExpressionEmitter(
            else_block, self.bb_names, self.var_names,
            self.local_env, self.global_env)
        else_emitter.visit(conditional.else_expr)

        else_result_instr = bytecode.CopyInst(result_var, else_emitter.result)
        else_emitter.end_block.add_inst(else_result_instr)

        new_end_block = bytecode.BasicBlock(next(self.bb_names))
        then_emitter.end_block.add_inst(bytecode.JmpInst(new_end_block))
        else_emitter.end_block.add_inst(bytecode.JmpInst(new_end_block))
        self.end_block = new_end_block

        self.result = result_var

    def visit_SNum(self, num: sexp.SNum) -> None:
        self.result = bytecode.NumLit(num)

    def visit_SBool(self, sbool: sexp.SBool) -> None:
        self.result = bytecode.BoolLit(sbool)

    def visit_SSym(self, sym: sexp.SSym) -> None:
        if self.quoted:
            self.result = bytecode.SymLit(sym)
        elif sym in self.local_env:
            self.result = self.local_env[sym]
        else:
            dest_var = bytecode.Var(next(self.var_names))
            lookup_instr = bytecode.LookupInst(dest_var, bytecode.SymLit(sym))
            self.parent_block.add_inst(lookup_instr)
            self.result = dest_var

    def visit_SVect(self, vect: sexp.SVect) -> None:
        var = bytecode.Var(next(self.var_names))
        instr = bytecode.AllocInst(
            var, bytecode.NumLit(sexp.SNum(len(vect.items))))
        self.parent_block.add_inst(instr)
        self.result = var

        parent_block = self.parent_block
        for (i, expr) in enumerate(vect.items):
            expr_emitter = ExpressionEmitter(
                parent_block, self.bb_names, self.var_names,
                self.local_env, self.global_env)
            expr_emitter.visit(expr)
            parent_block = expr_emitter.end_block

            store = bytecode.StoreInst(
                var, bytecode.NumLit(sexp.SNum(i)), expr_emitter.result)
            parent_block.add_inst(store)

    def visit_Quote(self, quote: sexp.Quote) -> None:
        if isinstance(quote.expr, sexp.SSym):
            self.quoted = True
            super().visit(quote.expr)
            return

        is_list = isinstance(quote.expr, sexp.SPair) and quote.expr.is_list
        assert quote.expr is sexp.Nil or is_list

        quoted_exprs = list(cast(sexp.SList, quote.expr))
        nil_var = bytecode.Var(next(self.var_names))
        nil_alloc = bytecode.AllocInst(
            nil_var, bytecode.NumLit(sexp.SNum(0)))

        cdr = nil_var
        self.parent_block.add_inst(nil_alloc)

        for expr in reversed(quoted_exprs):
            pair_var = bytecode.Var(next(self.var_names))
            pair_alloc = bytecode.AllocInst(
                pair_var, bytecode.NumLit(sexp.SNum(2)))
            self.parent_block.add_inst(pair_alloc)

            expr_emitter = ExpressionEmitter(
                self.parent_block, self.bb_names, self.var_names,
                self.local_env, self.global_env, quoted=True)
            expr_emitter.visit(expr)

            store_car = bytecode.StoreInst(
                pair_var, bytecode.NumLit(sexp.SNum(0)), expr_emitter.result)
            self.parent_block.add_inst(store_car)

            store_cdr = bytecode.StoreInst(
                pair_var, bytecode.NumLit(sexp.SNum(1)), cdr
            )
            self.parent_block.add_inst(store_cdr)

            cdr = pair_var

        self.result = cdr

    def visit_SCall(self, call: sexp.SCall) -> None:
        assert not self.quoted, 'Non-primitives in quoted list unsupported'

        func_expr_emitter = ExpressionEmitter(
            self.parent_block, self.bb_names, self.var_names,
            self.local_env, self.global_env)
        func_expr_emitter.visit(call.func)

        args: List[bytecode.Parameter] = []
        arg_emitter: Optional[ExpressionEmitter] = None
        self.end_block = func_expr_emitter.end_block
        for arg in call.args:
            arg_emitter = ExpressionEmitter(
                self.end_block, self.bb_names, self.var_names,
                self.local_env, self.global_env
            )
            arg_emitter.visit(arg)
            args.append(arg_emitter.result)
            self.end_block = arg_emitter.end_block

        call_result_var = bytecode.Var(next(self.var_names))
        call_instr = bytecode.CallInst(
            call_result_var, func_expr_emitter.result, args)

        if arg_emitter is None:
            func_expr_emitter.end_block.add_inst(call_instr)
        else:
            arg_emitter.end_block.add_inst(call_instr)

        self.result = call_result_var


def name_generator(prefix: str) -> Iterator[str]:
    count = 0
    while True:
        yield f'{prefix}{count}'
        count += 1
