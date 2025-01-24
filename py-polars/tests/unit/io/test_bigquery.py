# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path

import pytest

import polars as pl
from polars.io.bigquery import _predicate_to_row_restriction


class TestBigQueryExpressions:
    """Test coverage for `bigquery` expressions comprehension."""

    def test_is_null_expression(self) -> None:
        expr = pl.col("id").is_null()
        assert _predicate_to_row_restriction(expr) == "(`id` IS NULL)"

    def test_is_not_null_expression(self) -> None:
        expr = pl.col("id").is_not_null()
        assert _predicate_to_row_restriction(expr) == "(`id` IS NOT NULL)"

    def test_parse_combined_expression(self) -> None:
        from pyiceberg.expressions import (
            And,
            EqualTo,
            GreaterThan,
            In,
            Or,
            Reference,
            literal,
        )

        expr = _to_ast(
            "(((pa.compute.field('str') == '2') & (pa.compute.field('id') > 10)) | (pa.compute.field('id')).isin([1,2,3]))"
        )
        assert _convert_predicate(expr) == Or(
            left=And(
                left=EqualTo(term=Reference(name="str"), literal=literal("2")),
                right=GreaterThan(term="id", literal=literal(10)),
            ),
            right=In("id", {literal(1), literal(2), literal(3)}),
        )

    def test_parse_gt(self) -> None:
        from pyiceberg.expressions import GreaterThan

        expr = _to_ast("(pa.compute.field('ts') > '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThan("ts", "2023-08-08")

    def test_parse_gteq(self) -> None:
        from pyiceberg.expressions import GreaterThanOrEqual

        expr = _to_ast("(pa.compute.field('ts') >= '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThanOrEqual("ts", "2023-08-08")

    def test_parse_eq(self) -> None:
        from pyiceberg.expressions import EqualTo

        expr = _to_ast("(pa.compute.field('ts') == '2023-08-08')")
        assert _convert_predicate(expr) == EqualTo("ts", "2023-08-08")

    def test_parse_lt(self) -> None:
        from pyiceberg.expressions import LessThan

        expr = _to_ast("(pa.compute.field('ts') < '2023-08-08')")
        assert _convert_predicate(expr) == LessThan("ts", "2023-08-08")

    def test_parse_lteq(self) -> None:
        from pyiceberg.expressions import LessThanOrEqual

        expr = _to_ast("(pa.compute.field('ts') <= '2023-08-08')")
        assert _convert_predicate(expr) == LessThanOrEqual("ts", "2023-08-08")
