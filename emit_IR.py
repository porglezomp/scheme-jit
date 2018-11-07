from typing import Dict, Iterator, List, Optional

import bytecode
import scheme
from environment import Environment
from visitor import Visitor


class FunctionEmitter(Visitor):
    def __init__(self, global_env: Dict[scheme.SSym, scheme.SExp]) -> None:
        self.global_env = global_env
        # self.functions: List[bytecode.Function] = []

    def visit_SFunction(self, func: scheme.SFunction) -> None:
        # super().visit_SFunction(func)

        local_env: Dict[scheme.SSym, bytecode.Var] = {}
        for param in func.params:
            local_env[param] = bytecode.Var(param.name)

        bb_names = name_generator('bb')
        var_names = name_generator('var')

        body_exprs = list(func.body)

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
        # self.functions.append(emitted_func)


class ExpressionEmitter(Visitor):
    result: bytecode.Parameter

    def __init__(self, parent_block: bytecode.BasicBlock,
                 bb_names: Iterator[str],
                 var_names: Iterator[str],
                 local_env: Dict[scheme.SSym, bytecode.Var],
                 global_env: Dict[scheme.SSym, scheme.SExp],
                 quoted: bool = False) -> None:
        self.parent_block = parent_block
        self.end_block = parent_block
        self.bb_names = bb_names
        self.var_names = var_names
        self.local_env = local_env
        self.global_env = global_env
        self.quoted = quoted

    def visit_SFunction(self, func: scheme.SFunction) -> None:
        assert func.is_lambda, 'Encountered nested named function'

        func_emitter = FunctionEmitter(self.global_env)
        func_emitter.visit(func)

        lambda_var = bytecode.Var(next(self.var_names))
        lookup_lambda_instr = bytecode.LookupInst(
            lambda_var, bytecode.SymLit(func.name))

        self.parent_block.add_inst(lookup_lambda_instr)
        self.result = lambda_var

    def visit_SNum(self, num: scheme.SNum) -> None:
        self.result = bytecode.NumLit(num)

    def visit_SBool(self, sbool: scheme.SBool) -> None:
        self.result = bytecode.BoolLit(sbool)

    def visit_SSym(self, sym: scheme.SSym) -> None:
        if self.quoted:
            self.result = bytecode.SymLit(sym)
        elif sym in self.local_env:
            self.result = self.local_env[sym]
        elif sym in self.global_env:
            dest_var = bytecode.Var(next(self.var_names))
            lookup_instr = bytecode.LookupInst(dest_var, bytecode.SymLit(sym))
            self.parent_block.add_inst(lookup_instr)
            self.result = dest_var
        else:
            raise EnvBindingNotFound(f'Name not found: "{sym.name}"')

    def visit_SVect(self, vect: scheme.SVect) -> None:
        var = bytecode.Var(next(self.var_names))
        instr = bytecode.AllocInst(
            var, bytecode.NumLit(scheme.SNum(len(vect.items))))
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
                var, bytecode.NumLit(scheme.SNum(i)), expr_emitter.result)
            parent_block.add_inst(store)

    def visit_Quote(self, quote: scheme.Quote) -> None:
        quoted_exprs = list(quote.slist)
        nil_var = bytecode.Var(next(self.var_names))
        nil_alloc = bytecode.AllocInst(
            nil_var, bytecode.NumLit(scheme.SNum(0)))

        cdr = nil_var
        self.parent_block.add_inst(nil_alloc)

        for expr in reversed(quoted_exprs):
            pair_var = bytecode.Var(next(self.var_names))
            pair_alloc = bytecode.AllocInst(
                pair_var, bytecode.NumLit(scheme.SNum(2)))
            self.parent_block.add_inst(pair_alloc)

            expr_emitter = ExpressionEmitter(
                self.parent_block, self.bb_names, self.var_names,
                self.local_env, self.global_env, quoted=True)
            expr_emitter.visit(expr)

            store_car = bytecode.StoreInst(
                pair_var, bytecode.NumLit(scheme.SNum(0)), expr_emitter.result)
            self.parent_block.add_inst(store_car)

            store_cdr = bytecode.StoreInst(
                pair_var, bytecode.NumLit(scheme.SNum(1)), cdr
            )
            self.parent_block.add_inst(store_cdr)

            cdr = pair_var

        self.result = cdr

    def visit_SCall(self, call: scheme.SCall) -> None:
        func_expr_emitter = ExpressionEmitter(
            self.parent_block, self.bb_names, self.var_names,
            self.local_env, self.global_env)
        func_expr_emitter.visit(call.func)

        args: List[bytecode.Parameter] = []
        arg_emitter: Optional[ExpressionEmitter] = None
        for arg in call.args:
            arg_emitter = ExpressionEmitter(
                func_expr_emitter.end_block, self.bb_names, self.var_names,
                self.local_env, self.global_env
            )
            arg_emitter.visit(arg)
            args.append(arg_emitter.result)

        call_result_var = bytecode.Var(next(self.var_names))
        call_instr = bytecode.CallInst(
            call_result_var, func_expr_emitter.result, args)

        if arg_emitter is None:
            func_expr_emitter.end_block.add_inst(call_instr)
        else:
            arg_emitter.end_block.add_inst(call_instr)

        self.result = call_result_var

    # def visit_SExp(self, expr: scheme.SExp) -> None:
    #     super().visit_SExp(expr)
    #     if expr == return_expr:
    #         if isinstance(final_instr, bytecode.Parameter):

    # def visit_SConditional(self, conditional: scheme.SConditional) -> None:
    #     then_block = bytecode.BasicBlock(next(self.bb_names), [])
    #     then_visitor = ExpressionEmitter(then_block, bb_names)
    #     then_visitor.visit(conditional.then_expr)

    #     self.parent_block.add_inst(bytecode.BrInst(, then_block))

    #     super().visit_SExp()


def name_generator(prefix: str) -> Iterator[str]:
    count = 0
    while True:
        yield f'{prefix}{count}'
        count += 1


class EnvBindingNotFound(Exception):
    """
    An exception that indicates that a requested symbol does not
    exist in an Environment.
    """
    pass
