from __future__ import annotations

import os
import re

from flask import Flask, render_template, request
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

app = Flask(__name__)

# Símbolos globales
x, y = sp.symbols("x y", real=True)
C = sp.Symbol("C")

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # 3x, xsin(y), 300x^2y
    convert_xor,                          # ^ como potencia
    function_exponentiation,
)

SAFE_LOCALS = {
    "x": x,
    "y": y,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "pi": sp.pi,
    "E": sp.E,
    "Abs": sp.Abs,
}

class ExactnessError(Exception):
    def __init__(self, message: str, dM_dy: sp.Expr, dN_dx: sp.Expr):
        super().__init__(message)
        self.dM_dy = dM_dy
        self.dN_dx = dN_dx

def latex(expr: sp.Expr) -> str:
    return sp.latex(sp.simplify(expr))

def parse(user_text: str) -> sp.Expr:
    """
    Parser amigable (tipo cuaderno):
    - Acepta multiplicación implícita: 300x^2y, xcos(y), ysen(x)
    - Acepta 'sen(...)' como 'sin(...)'
    - Acepta e^x o e^(...) -> exp(x) o exp(...)
    - Potencias: ^ (recomendado) o **
    """
    txt = (user_text or "").strip()
    if not txt:
        raise ValueError("Entrada vacía.")

    # quitar espacios
    txt = txt.replace(" ", "")

    # permitir ** o ^ (preferimos ^ para SymPy con convert_xor)
    txt = txt.replace("**", "^")

    # sen(...) -> sin(...)
    txt = re.sub(r"(?i)sen(?=\()", "sin", txt)

    # e^(...) -> exp(...)
    txt = re.sub(r"(?i)e\^\(([^)]+)\)", r"exp(\1)", txt)

    # e^x, e^sin(x), e^2x, e^x2  (hasta que encuentre un operador)
    def _exp_repl(m):
        inner = m.group(1)
        return f"exp({inner})"

    txt = re.sub(r"(?i)e\^([A-Za-z0-9_]+)", _exp_repl, txt)

    try:
        return sp.simplify(
            parse_expr(
                txt,
                local_dict=SAFE_LOCALS,
                transformations=TRANSFORMS,
                evaluate=True,
            )
        )
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la expresión: {e}")

def normalize_constant(expr: sp.Expr) -> sp.Expr:
    """
    Heurística: mostrar y = C e^{...} en vez de y = e^{C+...}
    """
    expr = sp.simplify(expr)
    try:
        expr = sp.factor(expr)
    except Exception:
        pass
    expr = expr.replace(sp.exp(C), C)
    expr = expr.replace(sp.E**C, C)
    return sp.simplify(expr)

def prefer_explicit_y(implicit_eq: sp.Equality) -> sp.Equality:
    """
    Intenta despejar y para evitar logaritmos cuando sea posible.
    """
    try:
        sols = sp.solve(implicit_eq, y)
    except Exception:
        return implicit_eq
    if not sols:
        return implicit_eq
    return sp.Eq(y, normalize_constant(sols[0]))

@app.get("/")
def index():
    return render_template("index.html")

@app.route("/separacion", methods=["GET", "POST"])
def separacion():
    resultado_latex = None
    pasos = None
    error = None

    if request.method == "POST":
        try:
            rhs = parse(request.form.get("rhs", ""))
            resultado_latex, pasos = resolver_separacion(rhs)
        except Exception as e:
            error = str(e)

    return render_template("separacion.html", resultado_latex=resultado_latex, pasos=pasos, error=error)

@app.route("/exactas", methods=["GET", "POST"])
def exactas():
    resultado_latex = None
    pasos = None
    error = None
    validacion = None

    if request.method == "POST":
        try:
            M = parse(request.form.get("M", ""))
            N = parse(request.form.get("N", ""))
            resultado_latex, pasos = resolver_exacta(M, N)
        except ExactnessError as ex:
            error = str(ex)
            validacion = {
                "dM_dy": latex(ex.dM_dy),
                "dN_dx": latex(ex.dN_dx),
                "diff": latex(sp.simplify(ex.dM_dy - ex.dN_dx)),
            }
        except Exception as e:
            error = str(e)

    return render_template(
        "exactas.html",
        resultado_latex=resultado_latex,
        pasos=pasos,
        error=error,
        validacion=validacion,
    )

def resolver_separacion(rhs: sp.Expr):
    """
    dy/dx = f(x)g(y)
    """
    rhs = sp.simplify(rhs)

    fx, gy = sp.factor_terms(rhs).as_independent(y, as_Add=False)
    fx = sp.simplify(fx)
    gy = sp.simplify(gy)

    # intento extra
    if fx.has(y) or gy.has(x):
        fact = sp.factor(rhs)
        fx, gy = sp.factor_terms(fact).as_independent(y, as_Add=False)
        fx = sp.simplify(fx)
        gy = sp.simplify(gy)

    if fx.has(y) or gy.has(x):
        raise ValueError(
            "No se pudo separar como f(x)·g(y).\n"
            "Ejemplos válidos: y, x*y, sin(x)*y, (x^2+1)/y, exp(x)*sin(y)."
        )

    if sp.simplify(gy) == 0:
        raise ValueError("La parte g(y) quedó en 0. Revisa la ecuación.")

    lhs_int = sp.integrate(sp.simplify(1 / gy), y)
    rhs_int = sp.integrate(fx, x)

    implicit = sp.Eq(lhs_int, rhs_int + C)
    displayed = prefer_explicit_y(implicit)

    steps = [
        {"t": "Entrada", "m": sp.latex(sp.Eq(sp.Derivative(y, x), rhs))},
        {"t": "Identificar", "m": sp.latex(sp.Eq(sp.Symbol("f(x)"), fx)) + r"\quad " + sp.latex(sp.Eq(sp.Symbol("g(y)"), gy))},
        {"t": "Separar", "m": sp.latex(sp.Eq((1/gy)*sp.Symbol("dy"), fx*sp.Symbol("dx")))},
        {"t": "Integrar", "m": sp.latex(sp.Eq(sp.Integral(1/gy, y), sp.Integral(fx, x)))},
        {"t": "Resultado (implícito)", "m": sp.latex(implicit)},
    ]
    if displayed != implicit:
        steps.append({"t": "Forma equivalente (sin log)", "m": sp.latex(displayed)})

    return sp.latex(displayed), steps

def resolver_exacta(M: sp.Expr, N: sp.Expr):
    """
    M dx + N dy = 0 (exacta)
    """
    M = sp.simplify(M)
    N = sp.simplify(N)

    dM_dy = sp.diff(M, y)
    dN_dx = sp.diff(N, x)

    if sp.simplify(dM_dy - dN_dx) != 0:
        raise ExactnessError(
            "La ecuación NO es exacta porque \\(\\partial M/\\partial y \\neq \\partial N/\\partial x\\).",
            dM_dy=dM_dy,
            dN_dx=dN_dx,
        )

    psi_partial = sp.integrate(M, x)
    hprime = sp.simplify(N - sp.diff(psi_partial, y))
    h = sp.integrate(hprime, y)

    psi = sp.simplify(psi_partial + h)
    sol = sp.Eq(psi, C)

    steps = [
        {"t": "Ecuación", "m": sp.latex(sp.Eq(M*sp.Symbol("dx") + N*sp.Symbol("dy"), 0))},
        {"t": "Verificar", "m": sp.latex(sp.Eq(sp.diff(M, y), dM_dy)) + r"\quad " + sp.latex(sp.Eq(sp.diff(N, x), dN_dx))},
        {"t": "Integrar M en x", "m": r"\Psi(x,y)=\int M\,dx=" + sp.latex(psi_partial) + r"+h(y)"},
        {"t": "Encontrar h'(y)", "m": r"h'(y)=N-\frac{\partial}{\partial y}\left(\int M\,dx\right)=" + sp.latex(hprime)},
        {"t": "Integrar h'(y)", "m": r"h(y)=\int h'(y)\,dy=" + sp.latex(h)},
        {"t": "Solución", "m": sp.latex(sol)},
    ]
    return sp.latex(sol), steps

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
