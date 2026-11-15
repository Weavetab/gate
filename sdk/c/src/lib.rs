use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::slice;

use weavetab_gate::{Config, Gate, Input};

fn json_err(msg: &str) -> *mut c_char {
    let body = serde_json::json!({ "error": msg }).to_string();
    CString::new(body).unwrap_or_default().into_raw()
}

unsafe fn read_choices<'a>(ptr: *const *const c_char, len: usize) -> Result<Vec<&'a str>, String> {
    if ptr.is_null() || len == 0 {
        return Err("empty choices".into());
    }
    let slice = slice::from_raw_parts(ptr, len);
    let mut out = Vec::with_capacity(len);
    for &p in slice {
        if p.is_null() {
            return Err("null choice pointer".into());
        }
        let s = CStr::from_ptr(p)
            .to_str()
            .map_err(|_| "invalid utf-8 choice".to_string())?;
        out.push(s);
    }
    Ok(out)
}

#[no_mangle]
pub unsafe extern "C" fn wt_gate_new(cache_dir: *const c_char, threads: usize) -> *mut Gate {
    let mut cfg = Config::default();
    if !cache_dir.is_null() {
        if let Ok(s) = CStr::from_ptr(cache_dir).to_str() {
            if !s.is_empty() {
                cfg.cache_dir = Some(PathBuf::from(s));
            }
        }
    }
    if threads > 0 {
        cfg.threads = threads;
    }
    Box::into_raw(Box::new(Gate::new(cfg)))
}

#[no_mangle]
pub unsafe extern "C" fn wt_gate_free(gate: *mut Gate) {
    if !gate.is_null() {
        drop(Box::from_raw(gate));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wt_gate_decide_text(
    gate: *const Gate,
    text: *const c_char,
    choices: *const *const c_char,
    choices_len: usize,
) -> *mut c_char {
    if gate.is_null() || text.is_null() {
        return json_err("null pointer");
    }
    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return json_err("invalid utf-8 text"),
    };
    let list = match read_choices(choices, choices_len) {
        Ok(v) => v,
        Err(e) => return json_err(&e),
    };
    match (*gate).decide(Input::Text(text_str), &list) {
        Ok(d) => CString::new(serde_json::to_string(&d).unwrap_or_default())
            .unwrap_or_default()
            .into_raw(),
        Err(e) => json_err(&e.to_string()),
    }
}

#[no_mangle]
pub unsafe extern "C" fn wt_gate_decide_image(
    gate: *const Gate,
    image_bytes: *const u8,
    image_len: usize,
    choices: *const *const c_char,
    choices_len: usize,
) -> *mut c_char {
    if gate.is_null() || image_bytes.is_null() || image_len == 0 {
        return json_err("null or empty image");
    }
    let bytes = slice::from_raw_parts(image_bytes, image_len);
    let list = match read_choices(choices, choices_len) {
        Ok(v) => v,
        Err(e) => return json_err(&e),
    };
    match (*gate).decide(Input::ImageBytes(bytes), &list) {
        Ok(d) => CString::new(serde_json::to_string(&d).unwrap_or_default())
            .unwrap_or_default()
            .into_raw(),
        Err(e) => json_err(&e.to_string()),
    }
}

#[no_mangle]
pub unsafe extern "C" fn wt_gate_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}
