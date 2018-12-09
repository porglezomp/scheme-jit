from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, cast

import bytecode
import scheme_types
import sexp
from errors import EnvBindingNotFound
from find_tail_calls import TailCallData
from visitor import Visitor

IS_FUNCTION_TRAP = bytecode.BasicBlock(
    'non_function',
    [bytecode.TrapInst('Attempted to call a non-function')])
ARITY_TRAP = bytecode.BasicBlock(
    'wrong_arity',
    [bytecode.TrapInst('Call with the wrong number of arguments')])


class FunctionEmitter(Visitor):
    def __init__(self, global_env: Dict[sexp.SSym, sexp.Value],
                 tail_calls: Optional[List[TailCallData]] = None,
                 expr_types: Optional[scheme_types.FunctionTypeAnalyzer] = None
                 ) -> None:
        self._emitted_func: Optional[bytecode.Function] = None
        self.global_env = global_env
        self._tail_calls = tail_calls

        self._expr_types = expr_types

    def get_emitted_func(self) -> bytecode.Function:
        assert self._emitted_func is not None
        return self._emitted_func

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        local_env: Dict[sexp.SSym, bytecode.Var] = {}
        for param in func.params:
            local_env[param] = bytecode.Var(param.name)

        bb_names = name_generator('bb')
        var_names = name_generator('v')

        body_exprs = list(func.body)

        assert len(body_exprs) > 0, 'Function bodies must not be empty'

        start_block = bytecode.BasicBlock(next(bb_names))
        emitter = ExpressionEmitter(
            start_block, bb_names, var_names, local_env, self.global_env,
            function_entrypoint=start_block,
            tail_calls=self._tail_calls,
            expr_types=self._expr_types)
        for expr in body_exprs[:-1]:
            emitter.visit(expr)
            emitter = ExpressionEmitter(
                emitter.end_block, bb_names, var_names,
                local_env, self.global_env,
                function_entrypoint=start_block,
                tail_calls=self._tail_calls,
                expr_types=self._expr_types)

        emitter.visit(body_exprs[-1])

        return_instr = bytecode.ReturnInst(emitter.result)
        emitter.end_block.add_inst(return_instr)

        emitted_func = bytecode.Function(
            [bytecode.Var(param.name) for param in func.params],
            start_block
        )

        self._emitted_func = emitted_func


class ExpressionEmitter(Visitor):
    def __init__(self, parent_block: bytecode.BasicBlock,
                 bb_names: Iterator[str],
                 var_names: Iterator[str],
                 local_env: Dict[sexp.SSym, bytecode.Var],
                 global_env: Dict[sexp.SSym, sexp.Value],
                 *,
                 function_entrypoint: Optional[bytecode.BasicBlock] = None,
                 quoted: bool = False,
                 tail_calls: Optional[List[TailCallData]] = None,
                 expr_types: Optional[scheme_types.FunctionTypeAnalyzer] = None
                 ) -> None:
        self.parent_block = parent_block
        self.end_block = parent_block
        self.bb_names = bb_names
        self.var_names = var_names
        self.local_env = local_env
        self.global_env = global_env

        self._function_entrypoint = function_entrypoint
        self.quoted = quoted
        self._tail_calls = tail_calls
        self._expr_types = expr_types

        self._result: Optional[bytecode.Parameter] = None

    @property
    def result(self) -> bytecode.Parameter:
        assert self._result is not None, 'ExpressionEmitter has no result'
        return self._result

    @result.setter
    def result(self, val: bytecode.Parameter) -> None:
        self._result = val

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        assert func.is_lambda, 'Nested named functions not supported'
        assert not self.quoted, 'Non-primitives in quoted list unsupported'

        # Don't re-emit lambdas defined in a function we're specializing
        if not self._expr_types:
            func_emitter = FunctionEmitter(self.global_env)
            func_emitter.visit(func)
            func.code = func_emitter.get_emitted_func()

            assert func.name not in self.global_env, (
                f"Duplicate function name: {func.name}")
            self.global_env[func.name] = func

        lambda_var = bytecode.Var(next(self.var_names))
        lookup_lambda_instr = bytecode.LookupInst(
            lambda_var, bytecode.SymLit(func.name))

        self.parent_block.add_inst(lookup_lambda_instr)
        self.result = lambda_var

    def visit_SConditional(self, conditional: sexp.SConditional) -> None:
        assert not self.quoted, 'Non-primitives in quoted list unsupported'

        test_emitter = ExpressionEmitter(
            self.parent_block, self.bb_names, self.var_names,
            self.local_env, self.global_env,
            function_entrypoint=self._function_entrypoint,
            tail_calls=self._tail_calls,
            expr_types=self._expr_types)
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
            self.local_env, self.global_env,
            function_entrypoint=self._function_entrypoint,
            tail_calls=self._tail_calls,
            expr_types=self._expr_types)
        then_emitter.visit(conditional.then_expr)

        then_result_instr = bytecode.CopyInst(result_var, then_emitter.result)
        then_emitter.end_block.add_inst(then_result_instr)

        else_emitter = ExpressionEmitter(
            else_block, self.bb_names, self.var_names,
            self.local_env, self.global_env,
            function_entrypoint=self._function_entrypoint,
            tail_calls=self._tail_calls,
            expr_types=self._expr_types)
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
                self.local_env, self.global_env,
                function_entrypoint=self._function_entrypoint,
                tail_calls=self._tail_calls,
                expr_types=self._expr_types)
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
                self.local_env, self.global_env,
                quoted=True,
                function_entrypoint=self._function_entrypoint,
                tail_calls=self._tail_calls,
                expr_types=self._expr_types)
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

        if call.func == sexp.SSym('assert') and len(call.args) == 1:
            if self._is_true_type_assertion(call.args[0]):
                self.result = bytecode.NumLit(sexp.SNum(0))
                return

        is_tail_call = False
        tail_call_data = None
        if self._tail_calls is not None:
            is_tail_call = TailCallData(call) in self._tail_calls
            if is_tail_call:
                tail_call_data = self._tail_calls[
                    self._tail_calls.index(TailCallData(call))]

        is_known_function = False
        arity_known_correct = False
        call_args_incompatible = False
        if self._expr_types is not None:
            is_known_function = (
                self._expr_types.expr_type_known(call.func)
                and isinstance(self._expr_types.get_expr_type(call.func),
                               scheme_types.SchemeFunctionType)
            )
            if is_known_function:
                func_type = cast(scheme_types.SchemeFunctionType,
                                 self._expr_types.get_expr_type(call.func))
                arity_known_correct = func_type.arity == len(call.args)
            elif is_tail_call:
                assert tail_call_data is not None
                arity_known_correct = (
                    len(tail_call_data.get_func().params) == len(call.args))

            if arity_known_correct and is_tail_call:
                assert tail_call_data is not None
                for arg_expr, param in zip(call.args,
                                           tail_call_data.func_params):
                    arg_type = self._expr_types.get_expr_type(arg_expr)
                    param_type = self._expr_types.get_expr_type(param)
                    if not (arg_type < param_type):
                        call_args_incompatible = True

        func_expr_emitter = None
        optimize_tail_call = (
            is_tail_call
            and arity_known_correct
            and not call_args_incompatible)
        if not optimize_tail_call:
            func_expr_emitter = ExpressionEmitter(
                self.parent_block, self.bb_names, self.var_names,
                self.local_env, self.global_env,
                function_entrypoint=self._function_entrypoint,
                tail_calls=self._tail_calls,
                expr_types=self._expr_types)
            func_expr_emitter.visit(call.func)
            self.end_block = func_expr_emitter.end_block

            if not is_known_function:
                self._add_is_function_check(
                    func_expr_emitter.result, self.end_block)

            if not arity_known_correct:
                self._add_arity_check(
                    func_expr_emitter.result, self.end_block,
                    len(call.args))

        args: List[bytecode.Parameter] = []
        arg_emitter: Optional[ExpressionEmitter] = None
        for arg_expr in call.args:
            arg_emitter = ExpressionEmitter(
                self.end_block, self.bb_names, self.var_names,
                self.local_env, self.global_env,
                function_entrypoint=self._function_entrypoint,
                tail_calls=self._tail_calls,
                expr_types=self._expr_types
            )
            arg_emitter.visit(arg_expr)
            args.append(arg_emitter.result)
            self.end_block = arg_emitter.end_block

        if optimize_tail_call:
            assert tail_call_data is not None
            for arg, param in zip(args, tail_call_data.func_params):
                local_var = self.local_env[param]
                if arg != local_var:
                    self.end_block.add_inst(bytecode.CopyInst(local_var, arg))

            assert self._function_entrypoint is not None
            self.end_block.add_inst(
                bytecode.JmpInst(self._function_entrypoint))

            # We need a placeholder result since we're jumping back
            # to the beginning of the function
            self.result = bytecode.NumLit(sexp.SNum(0))
        else:
            assert func_expr_emitter is not None
            call_result_var = bytecode.Var(next(self.var_names))
            call_instr = bytecode.CallInst(
                call_result_var, func_expr_emitter.result, args)

            self.end_block.add_inst(call_instr)
            self.result = call_result_var

    def _is_true_type_assertion(self, assert_arg: sexp.SExp) -> bool:
        if self._expr_types is None:
            return False

        if not isinstance(assert_arg, sexp.SCall):
            return False

        if assert_arg.func not in self._TYPE_QUERIES:
            return False

        if len(assert_arg.args) != 1:
            return False

        type_query_arg = assert_arg.args[0]
        if not self._expr_types.expr_type_known(type_query_arg):
            return False

        return (self._expr_types.get_expr_type(type_query_arg)
                < self._TYPE_QUERIES[cast(sexp.SSym, assert_arg.func)])

    _TYPE_QUERIES: Dict[sexp.SSym, scheme_types.SchemeObjectType] = {
        sexp.SSym('number?'): scheme_types.SchemeNum,
        sexp.SSym('symbol?'): scheme_types.SchemeSym,
        sexp.SSym('vector?'): scheme_types.SchemeVectType(None),
        sexp.SSym('function?'): scheme_types.SchemeFunctionType(None),
        sexp.SSym('bool?'): scheme_types.SchemeBool,
        sexp.SSym('pair?'): scheme_types.SchemeVectType(2),
        sexp.SSym('nil?'): scheme_types.SchemeVectType(0),
    }

    def _add_is_function_check(
            self, function_expr: bytecode.Parameter,
            add_to_block: bytecode.BasicBlock) -> None:
        typeof_var = bytecode.Var(next(self.var_names))
        typeof_instr = bytecode.TypeofInst(
            typeof_var, function_expr)
        is_function_var = bytecode.Var(next(self.var_names))
        is_function_instr = bytecode.BinopInst(
            is_function_var, bytecode.Binop.SYM_EQ,
            typeof_var, bytecode.SymLit(sexp.SSym('function'))
        )
        branch_instr = bytecode.BrnInst(is_function_var, IS_FUNCTION_TRAP)

        for instr in [typeof_instr, is_function_instr, branch_instr]:
            add_to_block.add_inst(instr)

    def _add_arity_check(
            self, function_expr: bytecode.Parameter,
            add_to_block: bytecode.BasicBlock, arity: int) -> None:
        arity_var = bytecode.Var(next(self.var_names))
        add_to_block.add_inst(bytecode.ArityInst(arity_var, function_expr))
        correct_arity_var = bytecode.Var(next(self.var_names))
        add_to_block.add_inst(bytecode.BinopInst(
            correct_arity_var, bytecode.Binop.NUM_EQ,
            arity_var, bytecode.NumLit(sexp.SNum(arity))
        ))
        add_to_block.add_inst(bytecode.BrnInst(correct_arity_var, ARITY_TRAP))


def name_generator(prefix: str) -> Iterator[str]:
    count = 0
    while True:
        yield f'{prefix}{count}'
        count += 1
