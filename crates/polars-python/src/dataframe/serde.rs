use std::io::{BufReader, BufWriter, Cursor};
use std::ops::Deref;

use polars::prelude::*;
use polars_io::mmap::ReaderBytes;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use super::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::exceptions::ComputeError;
use crate::file::{get_file_like, get_mmap_bytes_reader};

#[pymethods]
impl PyDataFrame {
    #[cfg(feature = "ipc_streaming")]
    fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        // Used in pickle/pickling
        let mut buf: Vec<u8> = vec![];
        IpcStreamWriter::new(&mut buf)
            .with_compat_level(CompatLevel::newest())
            .finish(&mut self.df.clone())
            .expect("ipc writer");
        PyBytes::new(py, &buf)
    }

    #[cfg(feature = "ipc_streaming")]
    fn __setstate__(&mut self, state: &Bound<PyAny>) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<PyBackedBytes>() {
            Ok(s) => {
                let c = Cursor::new(&*s);
                let reader = IpcStreamReader::new(c);

                reader
                    .finish()
                    .map(|df| {
                        self.df = df;
                    })
                    .map_err(|e| PyPolarsErr::from(e).into())
            },
            Err(e) => Err(e),
        }
    }

    /// Serialize into binary data.
    fn serialize_binary(&self, py_f: PyObject) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        let writer = BufWriter::new(file);
        ciborium::into_writer(&self.df, writer)
            .map_err(|err| ComputeError::new_err(err.to_string()))
    }

    /// Serialize into a JSON string.
    #[cfg(feature = "json")]
    pub fn serialize_json(&mut self, py_f: PyObject) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self.df)
            .map_err(|err| ComputeError::new_err(err.to_string()))
    }

    /// Deserialize a file-like object containing binary data into a DataFrame.
    #[staticmethod]
    fn deserialize_binary(py_f: PyObject) -> PyResult<Self> {
        let file = get_file_like(py_f, false)?;
        let reader = BufReader::new(file);
        let df = ciborium::from_reader::<DataFrame, _>(reader)
            .map_err(|err| ComputeError::new_err(err.to_string()))?;
        Ok(df.into())
    }

    /// Deserialize a file-like object containing JSON string data into a DataFrame.
    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn deserialize_json(py: Python, py_f: Bound<PyAny>) -> PyResult<Self> {
        let mut mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;

        py.allow_threads(move || {
            let mmap_read: ReaderBytes = (&mut mmap_bytes_r).into();
            let bytes = mmap_read.deref();
            let df = serde_json::from_slice::<DataFrame>(bytes)
                .map_err(|err| ComputeError::new_err(err.to_string()))?;
            Ok(df.into())
        })
    }
}
