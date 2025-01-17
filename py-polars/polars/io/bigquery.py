from __future__ import annotations

import functools
import io
import json
from typing import TYPE_CHECKING, Sequence, Tuple, Iterator

from polars.datatypes import (
    Int64,
    String,
)
from polars.dependencies import bigquery, bigquery_storage_v1
import polars._reexport as pl
import polars.io.ipc
import polars.expr.meta


if TYPE_CHECKING:
    from polars import LazyFrame

def _bigquery_to_polars_type(field: bigquery.SchemaField):
    if field.mode.casefold() == "repeated":
        raise TypeError("array types not yet supported")

    type_ = field.field_type.casefold()
    if type_ in ("record", "struct"):
        raise TypeError("nested types not yet supported")

    if type_ == "string":
        return String()
    if type_ in ("integer", "int64"):
        return Int64()
    raise TypeError(f"got unexpected BigQuery type: {type_}")


def _bigquery_to_polars_types(table: bigquery.Table):
    """Convert BigQuery types to Polars types.
    
    Note: the REST API uses the names from the Legacy SQL data types (https://cloud.google.com/bigquery/docs/data-types).

    Also, the first request to the BigQuery Storage Read API provides an Arrow schema, but we want to delay starting a read session until after we know which columns and row filters we're using.
    """

    # TODO: if table is time partitioned, add pseudocolumn for _PARTITIONTIME to allow for partition filters. https://cloud.google.com/bigquery/docs/partitioned-tables#ingestion_time

    pl_schema = {}
    for field in table.schema:
        pl_schema[field.name] = _bigquery_to_polars_type(field)

    return pl_schema


def _json_expr_to_row_restriction(expr_json) -> str | None:
    """Create a row restriction to filter rows.
    
    Returns None if unknown operators are found and can't guarantee a superset of rows.
    """
    # TODO: iterative compilation to support deeper trees
    if 'BinaryExpr' in expr_json:
        binary_expr = expr_json['BinaryExpr']
        left = _json_expr_to_row_restriction(binary_expr['left'])
        if left is None:
            return None
        right = _json_expr_to_row_restriction(binary_expr['right'])
        if right is None:
            return None
        
        # TODO: lookup table instead of iterating through all possible types
        if binary_expr['op'] == 'And':
            return f"({left}) AND ({right})"

        if binary_expr['op']  == 'Eq':
            return f"{left} = {right}"
        
        return None

    if 'Column' in expr_json:
        return f"`{expr_json['Column']}`"
    
    if 'Literal' in expr_json:
        literal = expr_json['Literal']
        _, value = literal.popitem()
        return repr(value)
    
    return None


def _source_to_table_path_and_billing_project(source: str, *, default_project_id: str | None) -> Tuple[str, str]:
    """Converts source from BigQuery format project.dataset.table to a BigQuery Storage path."""
    parts = source.split(".")
    if len(parts) == 3:
        if default_project_id is not None:
            billing_project_id = default_project_id
        else:
            billing_project_id = parts[0]
        
        return f"projects/{parts[0]}/datasets/{parts[1]}/tables/{parts[2]}", billing_project_id
    elif len(parts) == 2:
        if default_project_id is None:
            raise ValueError(f"source {repr(source)} is missing project and no billing_project_id was set.")
        
        billing_project_id = default_project_id
        return f"projects/{default_project_id}/datasets/{parts[0]}/tables/{parts[1]}", billing_project_id
    
    raise ValueError(
        "expected 2 or 3 parts in the form of project.dataset.table "
        "(project optional if billing_project_id is set), but got "
        f"{len(parts)} parts in source: {repr(source)}."
    )


def scan_bigquery(source: str, *, bq_client = None, bqstorage_client = None, billing_project_id: str | None = None) -> LazyFrame:
    # TODO: customize auth and client_info
    if bq_client is None:
        bq_client = bigquery.Client(project=billing_project_id)

    if bqstorage_client is None:
        bqstorage_client = bigquery_storage_v1.BigQueryReadClient()

    table = bq_client.get_table(source)
    pl_schema = _bigquery_to_polars_types(table)

    table_path, billing_project_id = _source_to_table_path_and_billing_project(source, default_project_id=billing_project_id)
    func = functools.partial(_scan_bigquery_impl, bqstorage_client, table_path, billing_project_id)
    return pl.LazyFrame._scan_python_function(pl_schema, func, pyarrow=False)


def _scan_bigquery_impl(
        bqstorage_client,
        table_path: str,
        billing_project_id: str,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
        **kwargs,
    ) -> Iterator[pl.DataFrame]:
    """
    Generator function that creates the source.
    This function will be registered as IO source.
    """
    import google.cloud.bigquery_storage_v1.types as types

    read_request = types.CreateReadSessionRequest()
    read_session = types.ReadSession()
    read_options = types.ReadSession.TableReadOptions()

    if predicate is not None:
        predicate_expr = pl.Expr.deserialize(predicate)
        predicate_json_file = io.BytesIO()
        predicate_expr.meta.serialize(predicate_json_file, format="json")
        predicate_json_file.seek(0)
        predicate_json = json.load(predicate_json_file)
        read_options.row_restriction = _json_expr_to_row_restriction(predicate_json)
        print(read_options.row_restriction)

    read_options.selected_fields = with_columns
    read_session.read_options = read_options
    read_session.table = table_path
    read_session.data_format = types.DataFormat.ARROW
    
    read_request.parent = f"projects/{billing_project_id}"
    read_request.read_session = read_session

    # single-threaded for simplicity, consider increasing this to the number of parallel workers.
    read_request.max_stream_count = 1  

    # TODO: convert with_columns, predicate, n_rows, batch_size to request options
    session = bqstorage_client.create_read_session(read_request)
    arrow_schema = session.arrow_schema.serialized_schema

    if len(session.streams) == 0:
        # TODO: will arrow_schema be populated  in an empty read session?
        return

    reader = bqstorage_client.read_rows(session.streams[0].name)
    stream = io.BytesIO()
    stream.write(arrow_schema)
    
    for message in reader:
        stream.write(message.arrow_record_batch.serialized_record_batch)

    stream.seek(0)
    return polars.io.ipc.read_ipc_stream(stream)

    # TODO: filters that couldn't apply at the source