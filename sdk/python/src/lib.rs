use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use gate_core::{Config as CoreConfig, Gate as CoreGate, Input, OptionScore};

fn to_py_err(e: gate_core::GateError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn scores_to_py<'py>(py: Python<'py>, scores: &[OptionScore]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for s in scores {
        let d = PyDict::new(py);
        d.set_item("label", &s.label)?;
        d.set_item("score", s.score)?;
        list.append(d)?;
    }
    Ok(list)
}

fn run_with_input<T>(
    input: &Bound<'_, PyAny>,
    f: impl FnOnce(Input<'_>) -> Result<T, gate_core::GateError>,
) -> PyResult<T> {
    if let Ok(text) = input.extract::<String>() {
        return f(Input::Text(&text)).map_err(to_py_err);
    }
    if let Ok(bytes) = input.downcast::<PyBytes>() {
        return f(Input::ImageBytes(bytes.as_bytes())).map_err(to_py_err);
    }
    Err(PyValueError::new_err("input must be str (text) or bytes (image)"))
}

#[pyclass]
pub struct Gate {
    inner: CoreGate,
}

#[pymethods]
impl Gate {
    #[new]
    #[pyo3(signature = (cache_dir=None, cache_entries=4096, threads=2))]
    fn new(cache_dir: Option<String>, cache_entries: usize, threads: usize) -> Self {
        let cfg = CoreConfig {
            cache_dir: cache_dir.map(PathBuf::from),
            cache_entries,
            threads,
        };
        Self {
            inner: CoreGate::new(cfg),
        }
    }

    fn cache_path(&self) -> String {
        self.inner.cache_path().to_string_lossy().into_owned()
    }

    fn decide<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
        choices: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let refs: Vec<&str> = choices.iter().map(String::as_str).collect();
        let d = run_with_input(input, |inp| self.inner.decide(inp, &refs))?;
        let out = PyDict::new(py);
        out.set_item("choice", d.choice)?;
        out.set_item("confidence", d.confidence)?;
        out.set_item("scores", scores_to_py(py, &d.scores)?)?;
        out.set_item("cached", d.cached)?;
        out.set_item("latency_ms", d.latency_ms)?;
        Ok(out)
    }

    fn yes_no<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
        question: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let r = run_with_input(input, |inp| self.inner.yes_no(inp, question))?;
        let out = PyDict::new(py);
        out.set_item("answer", r.answer)?;
        out.set_item("probability", r.probability)?;
        out.set_item("cached", r.cached)?;
        out.set_item("latency_ms", r.latency_ms)?;
        Ok(out)
    }

    fn score<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
        levels: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let refs: Vec<&str> = levels.iter().map(String::as_str).collect();
        let s = run_with_input(input, |inp| self.inner.score(inp, &refs))?;
        let out = PyDict::new(py);
        out.set_item("level", s.level)?;
        out.set_item("expected", s.expected)?;
        out.set_item("scores", scores_to_py(py, &s.scores)?)?;
        out.set_item("cached", s.cached)?;
        out.set_item("latency_ms", s.latency_ms)?;
        Ok(out)
    }
}

#[pymodule]
fn weavetab_gate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Gate>()?;
    Ok(())
}
