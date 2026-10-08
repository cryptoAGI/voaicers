// SPDX-License-Identifier: MIT OR Apache-2.0
//! A small JSON reader and writer, in-crate (zero dependencies), for the files voaice.rs exchanges with the rest of
//! voaice: `.voaice` identities, forge events, `.persona` bundles.
//!
//! Two properties matter more than features:
//! - **Objects keep their insertion order.** voaice's canonical forms are order-sensitive: a vprint hashes
//!   `JSON.stringify` of an ordered map, so a reader that sorted keys would change the hash.
//! - **Numbers parse as the nearest f64** (Rust's `str::parse::<f64>` is correctly rounded, as are Python's `float()`
//!   and JavaScript's `Number`), so a metric read here is the same float the browser and Python hold.
//!
//! The writer emits the compact form (`,` and `:` with no spaces), which is what `JSON.stringify` and
//! `json.dumps(separators=(",", ":"), ensure_ascii=False)` produce for strings, integers and nested objects.
//! Floats are not written by the canonical paths (the vprint carries them as decimal strings), so no float
//! formatting rule is claimed here.

use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// the number as written, and its nearest f64; the text is kept so integers beyond 2^53 survive a round trip
    Num(String, f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn path(&self, keys: &[&str]) -> Option<&Value> {
        keys.iter().try_fold(self, |v, k| v.get(k))
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Value::Num(_, f) = self { Some(*f) } else { None }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }
    pub fn int(n: i128) -> Value {
        Value::Num(n.to_string(), n as f64)
    }
    pub fn obj(kv: Vec<(&str, Value)>) -> Value {
        Value::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// Compact JSON, keys in their stored order.
    pub fn to_compact(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }
    fn write(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Num(t, _) => out.push_str(t),
            Value::Str(s) => write_str(s, out),
            Value::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Value::Obj(kv) => {
                out.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// JSON string escaping as JSON.stringify and Python's json (ensure_ascii=False) do it: `"` `\` and the controls,
/// with \b \f \n \r \t short forms and \u00XX for the rest; everything else, non-ASCII included, as is.
fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{} at byte {}", what, self.i))
    }
    fn lit(&mut self, s: &str, v: Value) -> Result<Value, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            self.err("bad literal")
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > 128 {
            return self.err("nested too deep");
        }
        match self.b.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'n') => self.lit("null", Value::Null),
            Some(b't') => self.lit("true", Value::Bool(true)),
            Some(b'f') => self.lit("false", Value::Bool(false)),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Value::Arr(a));
                }
                loop {
                    self.ws();
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Value::Arr(a));
                        }
                        _ => return self.err("expected , or ]"),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Value::Obj(kv));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return self.err("expected a key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return self.err("expected :");
                    }
                    self.i += 1;
                    self.ws();
                    kv.push((k, self.value(depth + 1)?));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Value::Obj(kv));
                        }
                        _ => return self.err("expected , or }"),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => {
                let s = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.b[s..self.i]).unwrap();
                match t.parse::<f64>() {
                    Ok(f) => Ok(Value::Num(t.to_string(), f)),
                    Err(_) => self.err("bad number"),
                }
            }
            _ => self.err("unexpected character"),
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        let s = std::str::from_utf8(h).map_err(|_| "bad \\u escape")?;
        self.i += 4;
        u32::from_str_radix(s, 16).map_err(|_| "bad \\u escape".to_string())
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let s = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.b[s..self.i]).map_err(|_| "invalid UTF-8")?);
            match self.b.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => {
                    self.i += 1;
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut c = self.hex4()?;
                            if (0xD800..0xDC00).contains(&c) && self.b[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                c = 0x10000 + ((c - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3ff);
                            }
                            out.push(char::from_u32(c).unwrap_or('\u{fffd}'));
                        }
                        _ => return self.err("bad escape"),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_is_kept_and_compact_round_trips() {
        let t = r#"{"v":"dvscope/1","precision":18,"metrics":{"rms":{"wei":"1","decimal":"0.000000000000000001"},"a":[1,true,null]}}"#;
        let v = parse(t).unwrap();
        assert_eq!(v.to_compact(), t);
        assert_eq!(v.path(&["metrics", "rms", "wei"]).and_then(Value::as_str), Some("1"));
    }

    #[test]
    fn escapes_match_json_stringify() {
        let v = Value::str("a\"b\\c\nd\u{1}é😀");
        assert_eq!(v.to_compact(), "\"a\\\"b\\\\c\\nd\\u0001é😀\"");
        assert_eq!(parse(&v.to_compact()).unwrap(), v);
        assert_eq!(parse(r#""😀""#).unwrap(), Value::str("😀"));
    }

    #[test]
    fn numbers_are_the_nearest_f64() {
        let v = parse("[0.16626077485650595, 96.8048780487805, 1e-7, -3]").unwrap();
        if let Value::Arr(a) = v {
            assert_eq!(a[0].as_f64(), Some(0.16626077485650595));
            assert_eq!(a[1].as_f64(), Some(96.8048780487805));
            assert_eq!(a[3].as_f64(), Some(-3.0));
        }
    }
}
