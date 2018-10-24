import scheme
from scheme import SSym, SVect, SNum


def test_to_slist():
    assert scheme.to_slist([SSym(1), SSym(2), SSym(3)]) == SVect([
        SSym(1),
        SVect([
            SSym(2),
            SVect([
                SSym(3),
                SVect([]),
            ]),
        ]),
    ])


def test_parse_atoms():
    assert scheme.parse("hi") == [SSym("hi")]
    assert scheme.parse("hi hey hoi") == [
        SSym("hi"),
        SSym("hey"),
        SSym("hoi"),
    ]
    assert scheme.parse("42 foo") == [SNum(42), SSym("foo")]


def test_parse_list():
    assert scheme.parse("()") == [SVect([])]
    assert scheme.parse("(1 2 3)") == [
        scheme.to_slist([SNum(1), SNum(2), SNum(3)]),
    ]
    assert scheme.parse("(if (< x 0) 1 2)") == [
        scheme.to_slist([
            SSym("if"),
            scheme.to_slist([SSym("<"), SSym("x"), SNum(0)]),
            SNum(1),
            SNum(2),
        ]),
    ]


def test_vector():
    assert scheme.parse("[]") == [SVect([])]
    assert scheme.parse("[1 [2 [3 []]]]") == [
        SVect([SNum(1), SVect([SNum(2), SVect([SNum(3), SVect([])])])]),
    ]
