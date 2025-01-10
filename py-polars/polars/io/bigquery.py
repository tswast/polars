import functools
import io
from typing import TYPE_CHECKING, Tuple, Iterator

from polars.datatypes import (
    Int64,
    String,
)
from polars.dependencies import bigquery_storage_v1
import polars._reexport as pl


if TYPE_CHECKING:
    from polars import LazyFrame


def _bigquery_to_polars_types():
    """Convert BigQuery types to Polars types.
    
    Note: the REST API uses the names from the Legacy SQL data types (https://cloud.google.com/bigquery/docs/data-types).

    Also, the first request to the BigQuery Storage Read API provides an Arrow schema, but we want to delay starting a read session until after we know which columns and row filters we're using.
    """

    # TODO: don't hardcode the bigquery-public-data.usa_names.usa_1910_2013 table
    return {
        "state": String(),
        "gender": String(),
        "year": Int64(),
        "name": String(),
        "number": Int64(),
    }



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


def scan_bigquery(source: str, *, bqstorage_client = None, billing_project_id: str | None = None) -> LazyFrame:
    if bqstorage_client is None:
        bqstorage_client = bigquery_storage_v1.BigQueryReadClient()

    table_path, billing_project_id = _source_to_table_path_and_billing_project(source, default_project_id=billing_project_id)

    func = functools.partial(_scan_bigquery_impl, bqstorage_client, table_path, billing_project_id)
    pl_schema = _bigquery_to_polars_types()
    return pl.LazyFrame._scan_python_function(pl_schema, func, pyarrow=False)


def _scan_bigquery_impl(
        bqstorage_client,
        table_path: str,
        billing_project_id: str,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
    """
    Generator function that creates the source.
    This function will be registered as IO source.
    """
    import google.cloud.bigquery_storage_v1.types as types

    read_request = types.CreateReadSessionRequest()
    read_session = types.ReadSession()
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
        yield None  # How do we create an empty frame with just the requested columns?
        return

    reader = bqstorage_client.read_rows(session.streams[0].name)
    
    for message in reader:
        stream = io.BytesIO(arrow_schema)
        
        # TODO: do I need to make this respect the batch_size somehow?
        stream.write(message.arrow_record_batch.serialized_record_batch)
        stream.seek(0)
        yield pl.read_ipc_stream(stream)
