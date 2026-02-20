# PR Review: Add mangling support for noexcept(expr) in compound requirements

## Summary

This PR extends the Itanium name mangling and demangling to support `noexcept(expr)` (conditional noexcept) in compound requirements, as part of the P3822 proposal. The approach introduces a new `C <expression>` mangling production alongside the existing `N` marker for plain `noexcept`, which is a reasonable design that preserves backward compatibility.

## Issues

### Bug: Missing null check after `parseExpr()` for noexcept condition

**Files:** `llvm/include/llvm/Demangle/ItaniumDemangle.h`, `libcxxabi/src/demangle/ItaniumDemangle.h`

In the demangler, when `C` is consumed, `parseExpr()` is called but the result is never checked for `nullptr`:

```cpp
if (!Noexcept && consumeIf('C')) {
    Noexcept = true;
    NoexceptCond = getDerived().parseExpr();
    // Missing: if (NoexceptCond == nullptr) return nullptr;
}
```

If the expression is malformed, `NoexceptCond` will remain `nullptr`, and the code will silently produce a node that looks like plain `noexcept` instead of reporting a parse failure. Every other `parseExpr()` call in this function has a null check. This should be:

```cpp
if (!Noexcept && consumeIf('C')) {
    Noexcept = true;
    NoexceptCond = getDerived().parseExpr();
    if (NoexceptCond == nullptr)
        return nullptr;
}
```

### Bug: `match()` function not updated

**Files:** `llvm/include/llvm/Demangle/ItaniumDemangle.h`, `libcxxabi/src/demangle/ItaniumDemangle.h`

The `match` function on `ExprRequirement` still passes three arguments:

```cpp
template <typename Fn> void match(Fn F) const {
    F(Expr, IsNoexcept, TypeConstraint);
}
```

Per the documented contract at line 236-238 of the same file:
> "Call F with arguments that, when passed to the constructor of this node, would construct an equivalent node."

Since the constructor now takes `(Expr_, IsNoexcept_, NoexceptConstraint_, TypeConstraint_)`, `match` should be updated to:

```cpp
template <typename Fn> void match(Fn F) const {
    F(Expr, IsNoexcept, NoexceptConstraint, TypeConstraint);
}
```

This affects `DumpVisitor` (used in `llvm/lib/Demangle/ItaniumDemangle.cpp:260`) and any other downstream code that relies on `match` for visitation/reconstruction.

### Test coverage: plain `noexcept` + return type constraint test case lost

**File:** `clang/test/CodeGenCXX/mangle-requires.cpp`

The existing test case:
```cpp
{T() * 2} noexcept -> SmallerThan<1234>;
```
is replaced by:
```cpp
{T() * 2} noexcept(sizeof(T) == 1) -> SmallerThan<1234>;
```

This loses test coverage for the combination of plain `noexcept` with a return type constraint. The `{T() - 1} noexcept;` case only tests `noexcept` without a return type constraint. Consider keeping the old test case and adding the new one alongside it (or as a separate template).

### Nit: Grammar comment could be clearer

**Files:** `llvm/include/llvm/Demangle/ItaniumDemangle.h`, `libcxxabi/src/demangle/ItaniumDemangle.h`

The current comment shows the two productions separately:
```
// <requirement> ::= X <expression> [N] [R <type-constraint>]
//               ::= X <expression> C <expression> [R <type-constraint>]
```

It would be clearer to express `N` and `C <expression>` as mutually exclusive alternatives in a single production:
```
// <requirement> ::= X <expression> [N | C <expression>] [R <type-constraint>]
```

### Nit: Consider additional demangling test cases

**Files:** `llvm/include/llvm/Testing/Demangle/DemangleTestCases.inc`, `libcxxabi/test/DemangleTestCases.inc`

Only one demangling test case is added. Consider also testing:
- `noexcept(expr)` combined with a return type constraint (`-> T`)
- The existing mangled symbol with the updated expected output (the mangle-requires.cpp test changes the mangled symbol but the old demangling test case entry is not updated to reflect that the old mangled name is now for a different source construct)

## Overall Assessment

The design is sound -- using `C` as a mangling marker for conditional noexcept is unambiguous at this grammar position and maintains backward compatibility with existing `N` manglings. The two bugs above (missing null check and stale `match` function) should be addressed before merging.
