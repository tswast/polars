from typing import TYPE_CHECKING, Tuple

from polars.dependencies import bigquery_storage_v1


if TYPE_CHECKING:
    from polars import LazyFrame


def _bigquery_to_polars_types():
    pass


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
    import google.cloud.bigquery_storage_v1.types as types

    if bqstorage_client is None:
        bqstorage_client = bigquery_storage_v1.BigQueryReadClient()

    table_path, billing_project_id = _source_to_table_path_and_billing_project(source, default_project_id=billing_project_id)
    read_request = types.CreateReadSessionRequest()
    read_session = types.ReadSession()
    read_session.table = table_path
    read_session.data_format = types.DataFormat.ARROW
    read_request.parent = f"projects/{billing_project_id}"
    read_request.read_session = read_session

    # single-threaded for simplicity, consider increasing this to the number of parallel workers.
    read_request.max_stream_count = 1  

    # TODO: wait to do this until we know what predicates we need tp push down. 
    session = bqstorage_client.create_read_session(read_request)

    # TODO: test this. would there be a schema on an empty table?
    if len(session.streams) == 0:
        return LazyFrame()  # How do we create an empty frame?

    reader = bqstorage_client.read_rows(session.streams[0].name)